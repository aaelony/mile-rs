//! `mile-rs` — command-line entry point.
//!
//! Usage:
//!   mile-rs --config experiment.toml
//!   mile-rs --config experiment.yaml   # YAML accepted; a .toml equivalent is written
//!   mile-rs --config experiment.toml --seed 1234
//!
//! The config file is a TOML- (or YAML-) serialised `MileConfig`.
//! See `src/config.rs` for field documentation.

use std::{path::PathBuf, sync::Arc};

use burn::tensor::backend::AutodiffBackend;
use clap::Parser;

use mile_rs::{
    config::{BackendKind, MileConfig},
    cpu_device, gpu_device,
    dataset::TabularDataset,
    inference::diagnostics,
    models::params::FcnParamFlattener,
    probabilistic::BnnLogPosterior,
    sampling::ensemble::run_chains,
    training::train_warmstart,
    CpuBackend, GpuBackend, MileError,
};

const TOML_HELP: &str = "\
EXPERIMENT CONFIG FILE
======================
The config file is TOML (preferred) or YAML.  When a YAML file is supplied,
an equivalent .toml is written beside it automatically.

Minimal regression example
---------------------------
  experiment_name = \"my_experiment\"
  seed            = 42
  backend         = \"cpu\"           # \"cpu\" (parallel Rayon) | \"gpu\" (sequential Wgpu)
  output_dir      = \"results/my_experiment\"
  task            = \"regression\"

  [data]
  path          = \"data/train.tsv\"  # last column is the label; comma or whitespace delimited
  use_synthetic = false             # set true (and omit path) to use synthetic Gaussian data
  normalize     = true              # z-score features
  train_split   = 0.7
  valid_split   = 0.1
  test_split    = 0.2
  # datapoint_limit = 5000          # cap rows (omit → use all)

  [model]
  input_dim        = 10            # number of input features
  hidden_structure = [64, 64, 1]  # layer widths; final entry = output dim
  use_bias         = true

  [warmstart]
  enabled       = true
  max_epochs    = 500
  batch_size    = 32              # omit for full-batch gradient descent
  learning_rate = 1e-3
  patience      = 20             # early-stopping patience (epochs)

  [sampler]
  sampler      = \"mclmc\"         # \"mclmc\" | \"nuts\"
  n_chains     = 4
  n_samples    = 1000
  n_thinning   = 1               # keep every nth draw (1 = no thinning)
  warmup_steps = 1000

  # MCLMC-specific
  step_size_init           = 0.005
  desired_energy_var_start = 5e-4
  desired_energy_var_end   = 1e-4
  trust_in_estimate        = 1.5
  num_effective_samples    = 100
  diagonal_preconditioning = false

  # NUTS-specific (ignored when sampler = \"mclmc\")
  nuts_max_depth     = 10
  nuts_target_accept = 0.80

  [sampler.prior]
  name = \"StandardNormal\"        # \"StandardNormal\" | \"Normal\" | \"Laplace\"
  # For Normal or Laplace add:
  # loc   = 0.0
  # scale = 1.0

Classification task
-------------------
  Replace the top-level task line with an inline table:

  task = { classification = { n_classes = 7 } }

  Set the final entry of hidden_structure to match n_classes:
  hidden_structure = [32, 7]

Backend / parallelism
---------------------
  backend = \"cpu\"   → NdArray backend; chains run in parallel via Rayon (one per core).
  backend = \"gpu\"   → Wgpu backend;   chains run sequentially (avoids GPU queue contention).

  GPU is beneficial for large networks (many params, large batch).
  For small networks the CPU backend is usually faster end-to-end.

Seed override
-------------
  The seed in the config file is used by default.
  Pass --seed N on the command line to override it without editing the file:

    mile-rs --config experiment.toml --seed 123
";

/// Run a MILE (Microcanonical Langevin Ensemble) BNN experiment.
///
/// Loads an experiment config (TOML or YAML), optionally overrides the seed,
/// runs warmstart training and MCMC sampling, and writes samples to disk.
/// Pass --help for a full config file reference.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None, after_long_help = TOML_HELP)]
struct Cli {
    /// Path to the experiment config file (.toml or .yaml/.yml).
    /// YAML files are accepted and a .toml equivalent is written beside them.
    /// Omit to run with built-in defaults (useful for smoke-testing).
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Override the RNG seed from the config file.
    /// Chain i uses seed XOR i for independent randomness.
    #[arg(short, long, value_name = "N")]
    seed: Option<u64>,
}

fn main() -> Result<(), MileError> {
    // Default: mile_rs at info, everything else (wgpu, cubecl, etc.) at warn.
    // Override with RUST_LOG env var, e.g. RUST_LOG=debug for full output.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,mile_rs=info"),
    )
    .init();

    let cli = Cli::parse();
    let config_path = cli.config.as_deref();
    let seed_override = cli.seed;

    // ── Load (or default) config ──────────────────────────────────────────────
    let mut cfg: MileConfig = match config_path {
        Some(p) => {
            let content = std::fs::read_to_string(p)?;
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "toml" => toml::from_str(&content)?,
                "yaml" | "yml" => {
                    let parsed: MileConfig = serde_yaml::from_str(&content)?;
                    let toml_path = p.with_extension("toml");
                    if !toml_path.exists() {
                        match toml::to_string_pretty(&parsed) {
                            Ok(toml_str) => {
                                if let Err(e) = std::fs::write(&toml_path, toml_str) {
                                    log::warn!("Could not write TOML equivalent: {e}");
                                } else {
                                    log::info!(
                                        "YAML config detected — TOML equivalent written to {}",
                                        toml_path.display()
                                    );
                                }
                            }
                            Err(e) => log::warn!("Could not serialise config to TOML: {e}"),
                        }
                    }
                    parsed
                }
                other => {
                    return Err(MileError::Config(format!(
                        "Unknown config extension '.{other}'. Use .toml or .yaml"
                    )));
                }
            }
        }
        None => {
            log::warn!("No --config supplied; using built-in defaults");
            MileConfig::default()
        }
    };

    // ── Seed override ─────────────────────────────────────────────────────────
    if let Some(s) = seed_override {
        log::info!("Seed overridden by --seed flag: {s}");
        cfg.seed = s;
    }

    log::info!("Experiment: {}", cfg.experiment_name);
    log::info!("Backend:    {:?}", cfg.backend);
    log::info!("Sampler:    {:?}", cfg.sampler.sampler);
    log::info!("Chains:     {}", cfg.sampler.n_chains);
    log::info!("Samples:    {}", cfg.sampler.n_samples);
    log::info!("Seed:       {}", cfg.seed);

    // ── Dispatch to the right backend ─────────────────────────────────────────
    let t0 = std::time::Instant::now();

    let result = match cfg.backend {
        BackendKind::Cpu => run_experiment::<CpuBackend>(&cfg, config_path, cpu_device()),
        BackendKind::Gpu => run_experiment::<GpuBackend>(&cfg, config_path, gpu_device()),
    };

    let elapsed = t0.elapsed();
    let total_secs = elapsed.as_secs();
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    if h > 0 {
        eprintln!("\nTotal elapsed: {h}h {m:02}m {s:02}s");
    } else if m > 0 {
        eprintln!("\nTotal elapsed: {m}m {s:02}s");
    } else {
        eprintln!("\nTotal elapsed: {:.3}s", elapsed.as_secs_f64());
    }

    result
}

// ── Generic pipeline ──────────────────────────────────────────────────────────

fn run_experiment<B>(
    cfg: &MileConfig,
    config_path: Option<&std::path::Path>,
    device: B::Device,
) -> Result<(), MileError>
where
    B: AutodiffBackend,
    B::Device: Clone,
    BnnLogPosterior<B>: Send + Sync,
{
    let use_rayon = matches!(cfg.backend, BackendKind::Cpu);
    let output_dir = PathBuf::from(&cfg.output_dir);

    // ── Load data ─────────────────────────────────────────────────────────────
    let dataset = if cfg.data.use_synthetic {
        log::info!(
            "data.use_synthetic = true; generating synthetic Gaussian data \
             (n=1000, d={})",
            cfg.model.input_dim
        );
        let ds = synthetic_gaussian(1000, cfg.model.input_dim, cfg.seed);
        let synthetic_path = output_dir.join("data").join("train.csv");
        ds.to_csv(&synthetic_path)?;
        log::info!("Synthetic dataset written to {}", synthetic_path.display());
        ds
    } else {
        let p = cfg.data.path.as_deref().ok_or_else(|| {
            MileError::Config(
                "No data source configured. Set data.path or data.use_synthetic = true.".into(),
            )
        })?;
        let candidate = PathBuf::from(p);
        let data_path = if candidate.is_absolute() {
            candidate
        } else if let Some(config_dir) = config_path.and_then(|c| c.parent()) {
            config_dir.join(&candidate)
        } else {
            candidate
        };
        if !data_path.exists() {
            return Err(MileError::Config(format!(
                "Data file not found: {}",
                data_path.display()
            )));
        }
        log::info!("Loading data from {}", data_path.display());
        TabularDataset::from_csv(&data_path)?
    };

    let (train_ds, valid_ds, _test_ds) = dataset.split_train_valid_test(cfg.seed);

    let train_x = train_ds.features_tensor::<B>(&device);
    let train_y = train_ds.labels_tensor::<B>(&device);
    let valid_x = valid_ds.features_tensor::<B>(&device);
    let valid_y = valid_ds.labels_tensor::<B>(&device);

    let train_x_inner = train_x.clone().inner();
    let train_y_inner = train_y.clone().inner();

    // ── Warmstart ─────────────────────────────────────────────────────────────
    let init_positions: Vec<Vec<f32>> = if cfg.warmstart.enabled {
        log::info!("Starting warmstart training ({} chains)…", cfg.sampler.n_chains);
        train_warmstart::<B>(&cfg, &train_x, &train_y, &valid_x, &valid_y, &device)?
    } else {
        log::warn!("Warmstart disabled — sampling from random initialisations");
        let dim = FcnParamFlattener::from_config(&cfg.model).param_dim();
        (0..cfg.sampler.n_chains).map(|_| vec![0.0f32; dim]).collect()
    };

    log::info!("Init positions ready, param dim = {}", init_positions[0].len());

    // ── MCMC sampling ─────────────────────────────────────────────────────────
    let cfg_ref = Arc::new(cfg.clone());
    let train_x_ref = Arc::new(train_x_inner);
    let train_y_ref = Arc::new(train_y_inner);
    let device_ref = device.clone();

    log::info!("Starting {:?} sampling…", cfg.sampler.sampler);

    let model_factory = move || {
        BnnLogPosterior::<B>::new(
            &cfg_ref.model,
            &cfg_ref.sampler.prior,
            cfg_ref.task,
            (*train_x_ref).clone(),
            (*train_y_ref).clone(),
            device_ref.clone(),
        )
    };

    let results = run_chains(
        model_factory,
        &cfg.sampler,
        init_positions,
        cfg.seed,
        use_rayon,
        Some(&output_dir.join("samples")),
    )?;

    // ── Diagnostics ───────────────────────────────────────────────────────────
    let diag = diagnostics(&results);
    log::info!("{diag}");

    if diag.max_rhat > 1.01 {
        log::warn!("R-hat > 1.01 — chains may not have converged");
    }

    log::info!("Done. Samples written to {}", output_dir.join("samples").display());
    Ok(())
}

// ── Synthetic data helper ─────────────────────────────────────────────────────

fn synthetic_gaussian(n: usize, d: usize, seed: u64) -> TabularDataset {
    use rand::{rngs::SmallRng, RngExt, SeedableRng};
    use rand_distr::StandardNormal;
    let mut rng = SmallRng::seed_from_u64(seed);
    let features: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.sample(StandardNormal)).collect())
        .collect();
    let labels: Vec<f32> = features
        .iter()
        .map(|x| x[0] + rng.sample::<f32, _>(StandardNormal) * 0.1)
        .collect();
    TabularDataset { features, labels }
}
