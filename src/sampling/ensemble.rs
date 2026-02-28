//! Parallel chain runner — replaces `jax.pmap` from the Python implementation.
//!
//! MCLMC runs in two explicit phases:
//!   Phase 1 — all chains complete warmup (L and step_size adaptation).
//!   Phase 2 — all chains begin drawing samples simultaneously.
//!
//! No chain starts sampling until every chain has finished warmup.
//!
//! NUTS warmup is internal to nuts-rs, so NUTS chains run in a single phase.
//!
//! When `use_rayon = true` (CPU backend) each phase runs in parallel via Rayon.
//! When `use_rayon = false` (GPU backend) each phase runs sequentially.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::{
    config::{SamplerConfig, SamplerKind},
    probabilistic::LogPosterior,
    sampling::{
        mclmc::{
            kernel::mclmc_step,
            state::{MclmcParams, MclmcState},
            warmup::{find_l_and_step_size, WarmupConfig},
        },
        nuts::nuts_sample_chain,
    },
    MileError,
};

// ── Result types ──────────────────────────────────────────────────────────────

/// Samples from one chain.
#[derive(Debug)]
pub struct ChainResult {
    pub chain_id: usize,
    /// Each element is a flat parameter vector (one draw after thinning).
    pub samples: Vec<Vec<f32>>,
    /// Adapted MCLMC step size (MCLMC chains only).
    pub adapted_step_size: Option<f32>,
    /// Adapted MCLMC L (MCLMC chains only).
    pub adapted_l: Option<f32>,
}

/// Everything needed to start sampling after MCLMC warmup completes.
struct WarmupResult<M> {
    chain_id: usize,
    model: Arc<M>,
    state: MclmcState,
    params: MclmcParams,
    /// RNG advanced through warmup — continues into sampling for reproducibility.
    rng: SmallRng,
}

// ── Progress bar style ────────────────────────────────────────────────────────

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "chain {prefix:>2} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>5}/{len} ({percent}%) {msg}",
    )
    .unwrap()
    .progress_chars("=>-")
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run all chains and return their results.
///
/// For MCLMC: all chains complete warmup (phase 1) before any chain begins
/// sampling (phase 2).  For NUTS: warmup is internal to nuts-rs so chains
/// run in a single phase.
///
/// `use_rayon`: when `true` (CPU backend) each phase runs in parallel via Rayon;
///              when `false` (GPU backend) each phase runs sequentially.
pub fn run_chains<M, F>(
    model_factory: F,
    cfg: &SamplerConfig,
    init_positions: Vec<Vec<f32>>,
    seed: u64,
    use_rayon: bool,
    output_dir: Option<&Path>,
) -> Result<Vec<ChainResult>, MileError>
where
    M: LogPosterior + Send + Sync + 'static,
    F: Fn() -> M + Send + Sync,
{
    if let Some(dir) = output_dir {
        fs::create_dir_all(dir)?;
    }

    match cfg.sampler {
        SamplerKind::Mclmc => {
            run_mclmc_chains(model_factory, cfg, init_positions, seed, use_rayon, output_dir)
        }
        SamplerKind::Nuts => {
            run_nuts_chains(model_factory, cfg, init_positions, seed, use_rayon, output_dir)
        }
    }
}

// ── MCLMC: two-phase execution ────────────────────────────────────────────────

fn run_mclmc_chains<M, F>(
    model_factory: F,
    cfg: &SamplerConfig,
    init_positions: Vec<Vec<f32>>,
    seed: u64,
    use_rayon: bool,
    output_dir: Option<&Path>,
) -> Result<Vec<ChainResult>, MileError>
where
    M: LogPosterior + Send + Sync + 'static,
    F: Fn() -> M + Send + Sync,
{
    let n_chains = cfg.n_chains;
    let warmup_steps = cfg.warmup_steps as u64;
    let sample_steps = (cfg.n_samples * cfg.n_thinning) as u64;
    let style = bar_style();

    // Numbered 1..=n_chains
    let indexed: Vec<(usize, Vec<f32>)> = init_positions
        .into_iter()
        .enumerate()
        .map(|(i, pos)| (i + 1, pos))
        .collect();

    // ── Phase 1: Warmup ──────────────────────────────────────────────────────
    if use_rayon {
        log::info!("Phase 1 — warming up {} chains in parallel (CPU)", n_chains);
    } else {
        log::info!("Phase 1 — warming up {} chains sequentially (GPU)", n_chains);
    }

    let mp1 = Arc::new(MultiProgress::new());

    let warmup_fn = {
        let mp = Arc::clone(&mp1);
        let style = style.clone();
        let factory = &model_factory;
        move |(chain_id, init_pos): (usize, Vec<f32>)| -> Result<WarmupResult<M>, MileError> {
            let pb = mp.add(ProgressBar::new(warmup_steps));
            pb.set_style(style.clone());
            pb.set_prefix(chain_id.to_string());
            pb.set_message("warmup");

            let model = Arc::new(factory());
            let rng_seed = seed ^ (chain_id as u64);
            let mut rng = SmallRng::seed_from_u64(rng_seed);
            let logp_fn = |pos: &[f32]| model.value_and_grad(pos);
            let warmup_cfg = WarmupConfig::from_sampler(cfg);

            let (state, params) =
                find_l_and_step_size(init_pos, &warmup_cfg, &logp_fn, &mut rng, &pb)
                    .map_err(MileError::Logp)?;

            log::info!(
                "Chain {chain_id}: warmup done | step_size={:.4e} L={:.4e}",
                params.step_size,
                params.l
            );
            pb.finish_with_message("warmup done");

            Ok(WarmupResult { chain_id, model, state, params, rng })
        }
    };

    let warmed: Vec<WarmupResult<M>> = if use_rayon {
        indexed
            .into_par_iter()
            .map(warmup_fn)
            .collect::<Result<_, _>>()?
    } else {
        indexed
            .into_iter()
            .map(warmup_fn)
            .collect::<Result<_, _>>()?
    };

    // ── Barrier ──────────────────────────────────────────────────────────────
    eprintln!("\nAll {n_chains} chains warmed up — starting sampling\n");
    log::info!("All {n_chains} chains warmed up — starting sampling phase");

    // ── Phase 2: Sampling ────────────────────────────────────────────────────
    if use_rayon {
        log::info!("Phase 2 — sampling {} chains in parallel (CPU)", n_chains);
    } else {
        log::info!("Phase 2 — sampling {} chains sequentially (GPU)", n_chains);
    }

    let mp2 = Arc::new(MultiProgress::new());

    let sample_fn = {
        let mp = Arc::clone(&mp2);
        let style = style.clone();
        move |wr: WarmupResult<M>| -> Result<ChainResult, MileError> {
            let WarmupResult { chain_id, model, state, params, mut rng } = wr;

            let pb = mp.add(ProgressBar::new(sample_steps));
            pb.set_style(style.clone());
            pb.set_prefix(chain_id.to_string());
            pb.set_message("sampling");

            let logp_fn = |pos: &[f32]| model.value_and_grad(pos);
            let mut state = state;
            let mut samples: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_samples);
            let total_steps = cfg.n_samples * cfg.n_thinning;

            for step in 0..total_steps {
                let (next_state, _) =
                    mclmc_step(&state, &params, &logp_fn, &mut rng).map_err(MileError::Logp)?;
                state = next_state;
                if (step + 1) % cfg.n_thinning == 0 {
                    samples.push(state.position.clone());
                }
                pb.inc(1);
            }

            log::info!("Chain {chain_id}: drew {} samples", samples.len());
            pb.finish_with_message("done");

            let result = ChainResult {
                chain_id,
                samples,
                adapted_step_size: Some(params.step_size),
                adapted_l: Some(params.l),
            };
            if let Some(dir) = output_dir {
                save_chain(&result, dir)?;
            }
            Ok(result)
        }
    };

    if use_rayon {
        warmed
            .into_par_iter()
            .map(sample_fn)
            .collect::<Result<_, _>>()
    } else {
        warmed.into_iter().map(sample_fn).collect::<Result<_, _>>()
    }
}

// ── NUTS: single-phase execution ──────────────────────────────────────────────

fn run_nuts_chains<M, F>(
    model_factory: F,
    cfg: &SamplerConfig,
    init_positions: Vec<Vec<f32>>,
    seed: u64,
    use_rayon: bool,
    output_dir: Option<&Path>,
) -> Result<Vec<ChainResult>, MileError>
where
    M: LogPosterior + Send + Sync + 'static,
    F: Fn() -> M + Send + Sync,
{
    let n_chains = cfg.n_chains;
    let sample_steps = (cfg.n_samples * cfg.n_thinning) as u64;
    let style = bar_style();
    let mp = Arc::new(MultiProgress::new());

    if use_rayon {
        log::info!("Running {} NUTS chains in parallel (CPU)", n_chains);
    } else {
        log::info!("Running {} NUTS chains sequentially (GPU)", n_chains);
    }

    let run_one = {
        let mp = Arc::clone(&mp);
        let factory = &model_factory;
        move |(chain_id, init_pos): (usize, Vec<f32>)| -> Result<ChainResult, MileError> {
            let pb = mp.add(ProgressBar::new(sample_steps));
            pb.set_style(style.clone());
            pb.set_prefix(chain_id.to_string());
            pb.set_message("sampling");

            let model = Arc::new(factory());
            let rng_seed = seed ^ (chain_id as u64);
            let samples =
                nuts_sample_chain(model, cfg, chain_id, &init_pos, rng_seed, pb.clone())?;

            log::info!("Chain {chain_id}: drew {} samples", samples.len());
            pb.finish_with_message("done");

            let result = ChainResult {
                chain_id,
                samples,
                adapted_step_size: None,
                adapted_l: None,
            };
            if let Some(dir) = output_dir {
                save_chain(&result, dir)?;
            }
            Ok(result)
        }
    };

    let indexed: Vec<(usize, Vec<f32>)> = init_positions
        .into_iter()
        .enumerate()
        .map(|(i, pos)| (i + 1, pos))
        .collect();

    if use_rayon {
        indexed
            .into_par_iter()
            .map(run_one)
            .collect::<Result<_, _>>()
    } else {
        indexed
            .into_iter()
            .map(run_one)
            .collect::<Result<_, _>>()
    }
}

// ── Disk I/O ──────────────────────────────────────────────────────────────────

fn save_chain(result: &ChainResult, dir: &Path) -> Result<(), MileError> {
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;

    let path: PathBuf = dir.join(format!("chain_{}.json.gz", result.chain_id));
    let file = fs::File::create(&path)?;
    let mut gz = GzEncoder::new(file, Compression::default());
    let json = serde_json::to_string(&result.samples)?;
    gz.write_all(json.as_bytes()).map_err(MileError::Io)?;
    gz.finish().map_err(MileError::Io)?;
    log::info!("Chain {} saved to {}", result.chain_id, path.display());
    Ok(())
}
