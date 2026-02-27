//! Parallel chain runner — replaces `jax.pmap` from the Python implementation.
//!
//! When `use_rayon = true` (CPU backend) chains run in parallel via Rayon.
//! When `use_rayon = false` (GPU backend) chains run sequentially so all GPU
//! work stays on one thread and avoids contention on the command queue.

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
            warmup::{find_l_and_step_size, WarmupConfig},
        },
        nuts::nuts_sample_chain,
    },
    MileError,
};

// ── Result type ───────────────────────────────────────────────────────────────

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
/// `use_rayon`: when `true` (CPU backend) chains run in parallel via Rayon;
///              when `false` (GPU backend) chains run sequentially.
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

    let mp = MultiProgress::new();

    // MCLMC warmup runs separately before the sampling loop, so its steps must
    // be included in the bar total.  NUTS warmup is internal to nuts-rs and is
    // already embedded in the n_samples * n_thinning draw loop.
    let total_steps = match cfg.sampler {
        SamplerKind::Mclmc => (cfg.warmup_steps + cfg.n_samples * cfg.n_thinning) as u64,
        SamplerKind::Nuts => (cfg.n_samples * cfg.n_thinning) as u64,
    };

    let style = bar_style();

    // GPU runs chains sequentially — add a top-level counter so the user can
    // see overall progress across chains.  CPU (Rayon) shows N parallel bars
    // instead, which already conveys this information.
    let chain_bar: Arc<Option<ProgressBar>> = if !use_rayon {
        let cb_style = ProgressStyle::with_template(
            "overall   [{elapsed_precise}] {bar:40.green/black} {pos:>2}/{len} chains ({percent}%) {msg}",
        )
        .unwrap()
        .progress_chars("=>-");
        let cb = mp.insert(0, ProgressBar::new(cfg.n_chains as u64));
        cb.set_style(cb_style);
        cb.set_message("running");
        Arc::new(Some(cb))
    } else {
        Arc::new(None)
    };

    let run = {
        let chain_bar = Arc::clone(&chain_bar);
        move |(chain_id, init_pos): (usize, Vec<f32>)| {
            let pb = mp.add(ProgressBar::new(total_steps));
            pb.set_style(style.clone());
            pb.set_prefix(chain_id.to_string());
            pb.set_message("warmup");

            let model = Arc::new(model_factory());
            let rng_seed = seed ^ (chain_id as u64);
            let result = run_one_chain(&model, cfg, chain_id, init_pos, rng_seed, pb.clone())?;
            if let Some(dir) = output_dir {
                save_chain(&result, dir)?;
            }
            pb.finish_with_message("done");
            if let Some(ref cb) = *chain_bar {
                cb.inc(1);
                if cb.position() == cfg.n_chains as u64 {
                    cb.finish_with_message("done");
                }
            }
            Ok(result)
        }
    };

    let results: Vec<Result<ChainResult, MileError>> = if use_rayon {
        log::info!("Running {} chains in parallel (CPU backend)", cfg.n_chains);
        init_positions.into_par_iter().enumerate().map(run).collect()
    } else {
        log::info!("Running {} chains sequentially (GPU backend)", cfg.n_chains);
        init_positions.into_iter().enumerate().map(run).collect()
    };

    results.into_iter().collect()
}

// ── Per-chain dispatch ────────────────────────────────────────────────────────

fn run_one_chain<M: LogPosterior + 'static>(
    model: &Arc<M>,
    cfg: &SamplerConfig,
    chain_id: usize,
    init_pos: Vec<f32>,
    rng_seed: u64,
    pb: ProgressBar,
) -> Result<ChainResult, MileError> {
    match cfg.sampler {
        SamplerKind::Mclmc => run_mclmc_chain(model, cfg, chain_id, init_pos, rng_seed, pb),
        SamplerKind::Nuts => {
            let samples = nuts_sample_chain(Arc::clone(model), cfg, chain_id, &init_pos, pb)?;
            Ok(ChainResult {
                chain_id,
                samples,
                adapted_step_size: None,
                adapted_l: None,
            })
        }
    }
}

// ── MCLMC chain ───────────────────────────────────────────────────────────────

fn run_mclmc_chain<M: LogPosterior>(
    model: &Arc<M>,
    cfg: &SamplerConfig,
    chain_id: usize,
    init_pos: Vec<f32>,
    rng_seed: u64,
    pb: ProgressBar,
) -> Result<ChainResult, MileError> {
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    let logp_fn = |pos: &[f32]| model.value_and_grad(pos);

    // ── Warmup ────────────────────────────────────────────────────────────────
    // pb shows sub-phase messages and increments during warmup.
    let warmup_cfg = WarmupConfig::from_sampler(cfg);
    let (mut state, params) =
        find_l_and_step_size(init_pos, &warmup_cfg, &logp_fn, &mut rng, &pb)
            .map_err(MileError::Logp)?;

    log::info!(
        "Chain {chain_id}: warmup done | step_size={:.4e} L={:.4e}",
        params.step_size,
        params.l
    );

    // ── Sampling ──────────────────────────────────────────────────────────────
    pb.set_message("sampling");

    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_samples);
    let total_steps = cfg.n_samples * cfg.n_thinning;

    for step in 0..total_steps {
        let (next_state, _energy_change) =
            mclmc_step(&state, &params, &logp_fn, &mut rng).map_err(MileError::Logp)?;
        state = next_state;

        if (step + 1) % cfg.n_thinning == 0 {
            samples.push(state.position.clone());
        }

        pb.inc(1);
    }

    log::info!("Chain {chain_id}: drew {} samples", samples.len());

    Ok(ChainResult {
        chain_id,
        samples,
        adapted_step_size: Some(params.step_size),
        adapted_l: Some(params.l),
    })
}

// ── Disk I/O ──────────────────────────────────────────────────────────────────

fn save_chain(result: &ChainResult, dir: &Path) -> Result<(), MileError> {
    let path: PathBuf = dir.join(format!("chain_{}.json", result.chain_id));
    let json = serde_json::to_string(&result.samples)?;
    fs::write(&path, json)?;
    log::info!("Chain {} saved to {}", result.chain_id, path.display());
    Ok(())
}
