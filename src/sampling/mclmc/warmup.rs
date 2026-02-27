//! Three-phase MCLMC warmup: adapts `step_size` and `L`.
//!
//! Direct port of `src/training/warmup.py::mclmc_find_L_and_step_size`,
//! `make_L_step_size_adaptation`, `make_adaptation_L`, and `handle_nans`.
//!
//! # Phases
//!
//! | Phase | Steps                  | What happens                                   |
//! |-------|------------------------|------------------------------------------------|
//! | 1     | `tune1_steps`          | Step-size adaptation via energy-variance target |
//! | 2     | `tune2_steps`          | L estimation from position covariance; optional |
//! |       |                        | diagonal preconditioning                        |
//! | 3     | `tune3_steps`          | L refinement via ESS autocorrelation (FFT)      |

use indicatif::ProgressBar;
use rand::Rng;

use crate::{
    inference::effective_sample_size,
    probabilistic::LogpError,
    sampling::mclmc::{
        kernel::{init_chain, mclmc_step},
        state::{MclmcParams, MclmcState},
    },
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Mirrors the relevant fields of Python `SamplerConfig` for MCLMC warmup.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Phase 1: step-size adaptation steps.
    pub tune1_steps: usize,
    /// Phase 2: L + diagonal-cov estimation steps.
    pub tune2_steps: usize,
    /// Phase 3: L refinement via ESS (0 → skip).
    pub tune3_steps: usize,
    pub step_size_init: f32,
    pub desired_energy_var_start: f32,
    pub desired_energy_var_end: f32,
    pub trust_in_estimate: f32,
    pub num_effective_samples: usize,
    pub diagonal_preconditioning: bool,
}

impl WarmupConfig {
    /// Build from the sampler config, using the fixed 80/10/10 phase split.
    pub fn from_sampler(cfg: &crate::config::SamplerConfig) -> Self {
        let total = cfg.warmup_steps;
        Self {
            tune1_steps: (total as f32 * 0.8) as usize,
            tune2_steps: (total as f32 * 0.1) as usize,
            tune3_steps: (total as f32 * 0.1) as usize,
            step_size_init: cfg.step_size_init,
            desired_energy_var_start: cfg.desired_energy_var_start,
            desired_energy_var_end: cfg.desired_energy_var_end,
            trust_in_estimate: cfg.trust_in_estimate,
            num_effective_samples: cfg.num_effective_samples,
            diagonal_preconditioning: cfg.diagonal_preconditioning,
        }
    }
}

// ── Internal adaptation state ─────────────────────────────────────────────────

/// Running state for the step-size predictor (phase 1 + 2).
struct AdaptiveState {
    /// Exponential moving weight.
    time: f32,
    /// EMA of `xi / ε^6`.
    x_average: f32,
    /// Largest step size that has not caused a NaN / divergence.
    step_size_max: f32,
}

impl AdaptiveState {
    fn new() -> Self {
        Self {
            time: 0.0,
            x_average: 0.0,
            step_size_max: f32::INFINITY,
        }
    }
}

/// Running streaming average for position moments (used to estimate covariance).
struct StreamingAvg {
    weight: f32,
    /// Running mean of `x`.
    x_sum: Vec<f32>,
    /// Running mean of `x²`.
    x2_sum: Vec<f32>,
}

impl StreamingAvg {
    fn new(dim: usize) -> Self {
        Self {
            weight: 0.0,
            x_sum: vec![0.0; dim],
            x2_sum: vec![0.0; dim],
        }
    }

    /// Mirrors `blackjax.util.streaming_average_update`.
    fn update(&mut self, x: &[f32], weight: f32, zero_prevention: f32) {
        let new_weight = self.weight + weight;
        let denom = new_weight + zero_prevention;
        for ((xs, x2s), &xi) in self
            .x_sum
            .iter_mut()
            .zip(self.x2_sum.iter_mut())
            .zip(x.iter())
        {
            *xs = (self.weight * *xs + weight * xi) / denom;
            *x2s = (self.weight * *x2s + weight * xi * xi) / denom;
        }
        self.weight = new_weight;
    }

    /// Element-wise variance: `E[x²] - E[x]²`.
    fn variances(&self) -> Vec<f32> {
        self.x_sum
            .iter()
            .zip(self.x2_sum.iter())
            .map(|(&x, &x2)| (x2 - x * x).max(0.0))
            .collect()
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the three-phase MCLMC warmup and return tuned `(state, params)`.
///
/// Mirrors `mclmc_find_L_and_step_size` in `warmup.py`.
pub fn find_l_and_step_size(
    init_position: Vec<f32>,
    cfg: &WarmupConfig,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
    pb: &ProgressBar,
) -> Result<(MclmcState, MclmcParams), LogpError> {
    let dim = init_position.len();

    // Initialise chain state and hyperparameters.
    let mut state = init_chain(init_position, logp_fn, rng)?;
    let mut params = MclmcParams::initial(dim, cfg.step_size_init);

    // ── Phase 1 + 2: step-size + L adaptation ────────────────────────────────
    let (new_state, new_params, streaming) =
        l_step_size_adaptation(&state, &params, cfg, logp_fn, rng, pb)?;
    state = new_state;
    params = new_params;

    // Determine L from estimated parameter covariance.
    if cfg.tune2_steps > 0 {
        let variances = streaming.variances();
        let var_sum: f32 = variances.iter().sum();

        if cfg.diagonal_preconditioning {
            params.sqrt_diag_cov = variances.iter().map(|v| v.sqrt()).collect();
            params.l = (dim as f32).sqrt();

            // Extra fine-tuning steps after updating the preconditioner.
            let fine_steps = (cfg.tune2_steps / 3).max(1);
            let (fs, fp, _) =
                l_step_size_adaptation_steps(&state, &params, cfg, logp_fn, rng, fine_steps, true, pb)?;
            state = fs;
            params = fp;
        } else {
            params.l = var_sum.sqrt();
        }
    }

    // ── Phase 3: L refinement via ESS ────────────────────────────────────────
    if cfg.tune3_steps > 0 {
        let l_new = adaptation_l(&mut state, &params, cfg.tune3_steps, 0.4, logp_fn, rng, pb)?;
        params.l = l_new;
    }

    Ok((state, params))
}

// ── Phase 1+2 implementation ──────────────────────────────────────────────────

/// Run the combined step-size + covariance adaptation.
///
/// Returns `(final_state, final_params, streaming_avg)`.
fn l_step_size_adaptation(
    init_state: &MclmcState,
    init_params: &MclmcParams,
    cfg: &WarmupConfig,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
    pb: &ProgressBar,
) -> Result<(MclmcState, MclmcParams, StreamingAvg), LogpError> {
    let total = cfg.tune1_steps + cfg.tune2_steps;

    // Phase-2 mask: 0 during phase 1 (ignore samples), 1 during phase 2.
    // Mirrors Python: `mask = 1 - jnp.concatenate([zeros(tune1), ones(tune2)])`
    // Note: Python mask is inverted compared to the intuitive reading.
    // In Python, mask=1 means "ignore" (zero_prevention), mask=0 means "count".

    let (state, params, streaming) =
        l_step_size_adaptation_steps(init_state, init_params, cfg, logp_fn, rng, total, false, pb)?;

    Ok((state, params, streaming))
}

/// Inner loop shared by phase-1+2 and the fine-tuning pass after preconditioning.
fn l_step_size_adaptation_steps(
    init_state: &MclmcState,
    init_params: &MclmcParams,
    cfg: &WarmupConfig,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
    total_steps: usize,
    all_masked: bool,
    pb: &ProgressBar,
) -> Result<(MclmcState, MclmcParams, StreamingAvg), LogpError> {
    let dim = init_state.position.len();
    let decay_rate =
        (cfg.num_effective_samples as f32 - 1.0) / (cfg.num_effective_samples as f32 + 1.0);

    let mut state = init_state.clone();
    let mut params = init_params.clone();
    let mut adapt = AdaptiveState::new();
    let mut streaming = StreamingAvg::new(dim);

    // Set initial phase message.
    if all_masked {
        pb.set_message("warmup: fine-tune");
    } else {
        pb.set_message("warmup: step-size (1/3)");
    }

    for step in 0..total_steps {
        // Phase transition: step-size adaptation → L + covariance estimation.
        if !all_masked && step == cfg.tune1_steps {
            pb.set_message("warmup: L+cov (2/3)");
        }

        // mask = 1 → "ignore" this sample in streaming avg (phase-1 steps)
        // mask = 0 → "include" this sample (phase-2 steps)
        let mask = if all_masked || step < cfg.tune1_steps {
            1.0f32
        } else {
            0.0f32
        };

        let desired_energy_var = desired_energy_var_at(step, total_steps, cfg);

        let (new_state, new_params, success) = predictor(
            &state,
            &params,
            &mut adapt,
            logp_fn,
            rng,
            step,
            dim,
            desired_energy_var,
            cfg.trust_in_estimate,
            decay_rate,
        )?;

        // Update streaming average of position (for covariance estimation).
        let weight = (1.0 - mask) * (if success { 1.0 } else { 0.0 }) * new_params.step_size;
        streaming.update(&new_state.position, weight, mask);

        state = new_state;
        params = new_params;

        pb.inc(1);
    }

    Ok((state, params, streaming))
}

// ── Predictor (step-size adaptation) ─────────────────────────────────────────

/// One predictor step: advance the chain and update the step-size estimate.
///
/// Mirrors Python `predictor()` in `warmup.py:271-326`.
fn predictor(
    state: &MclmcState,
    params: &MclmcParams,
    adapt: &mut AdaptiveState,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
    _step: usize,
    dim: usize,
    desired_energy_var: f32,
    trust_in_estimate: f32,
    decay_rate: f32,
) -> Result<(MclmcState, MclmcParams, bool), LogpError> {
    // Advance one kernel step.
    let step_result = mclmc_step(state, params, logp_fn, rng);

    // Handle NaNs / divergences gracefully.
    let (success, next_state, step_size_max, energy_change) = match step_result {
        Ok((ns, ec)) => handle_nans(state, ns, params.step_size, adapt.step_size_max, ec),
        Err(LogpError::Recoverable(_)) => {
            // Treat a recoverable error like a NaN: shrink step size, stay put.
            let reduced = params.step_size * 0.8;
            (
                false,
                state.clone(),
                reduced.min(adapt.step_size_max),
                0.0,
            )
        }
        Err(e) => return Err(e),
    };

    adapt.step_size_max = step_size_max;

    // Compute weight and update EMA of `xi / ε^6`.
    let xi = energy_change.powi(2) / (dim as f32 * desired_energy_var) + 1e-8;
    let weight = (-0.5 * (xi.ln() / (6.0 * trust_in_estimate)).powi(2)).exp();

    adapt.x_average = decay_rate * adapt.x_average + weight * (xi / params.step_size.powi(6));
    adapt.time = decay_rate * adapt.time + weight;

    // New step size from `Var[E] = O(ε^6)`.
    let mut new_step_size = (adapt.x_average / adapt.time).powf(-1.0 / 6.0);
    new_step_size = new_step_size.min(adapt.step_size_max);

    let new_params = MclmcParams {
        step_size: new_step_size,
        l: params.l,
        sqrt_diag_cov: params.sqrt_diag_cov.clone(),
    };

    Ok((next_state, new_params, success))
}

// ── Phase 3: L refinement via ESS ────────────────────────────────────────────

/// Refine `L` using the effective sample size of collected samples.
///
/// Mirrors `make_adaptation_L` in `warmup.py:408-465`.
fn adaptation_l(
    state: &mut MclmcState,
    params: &MclmcParams,
    num_steps: usize,
    l_factor: f32,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
    pb: &ProgressBar,
) -> Result<f32, LogpError> {
    let dim = state.position.len();
    let max_params = 2000; // `fft_params_limit` from Python

    pb.set_message("warmup: ESS (3/3)");

    // Collect `num_steps` samples.
    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(num_steps);
    for _ in 0..num_steps {
        let (ns, _) = mclmc_step(state, params, logp_fn, rng)?;
        samples.push(ns.position.clone());
        *state = ns;
        pb.inc(1);
    }

    // Subsample parameter dimensions if the model is very large.
    let n_params = dim.min(max_params);
    let param_indices: Vec<usize> = (0..n_params).collect(); // first n_params dims

    // Compute ESS for each parameter dimension using the `mcmc` crate.
    let mean_ratio: f64 = param_indices
        .iter()
        .map(|&j| {
            let chain: Vec<f64> = samples.iter().map(|s| f64::from(s[j])).collect();
            let ess = effective_sample_size(&chain);
            if ess > 0.0 {
                num_steps as f64 / ess
            } else {
                1.0
            }
        })
        .sum::<f64>()
        / n_params as f64;

    Ok(l_factor * params.step_size * mean_ratio as f32)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Linearly (or exponentially) interpolate the desired energy variance.
///
/// Mirrors `get_desired_energy_var_linear` / `get_desired_energy_var_exp`.
fn desired_energy_var_at(step: usize, total: usize, cfg: &WarmupConfig) -> f32 {
    let progress = (step as f32 / total as f32).min(1.0);
    let start = cfg.desired_energy_var_start;
    let end = cfg.desired_energy_var_end;

    if start > 2.0 {
        // Exponential decay (used when start is very large).
        let tau = total as f32 / 4.0;
        start * (-(step as f32) / tau).exp() + end * (1.0 - (-(step as f32) / tau).exp())
    } else {
        // Linear decay.
        start - (start - end) * progress
    }
}

/// Handle NaN / divergence in the integrator output.
///
/// If the proposed position is non-finite, reject it (stay at previous state)
/// and shrink `step_size_max`.
///
/// Mirrors `handle_nans` in `warmup.py:468-483`.
///
/// Returns `(success, accepted_state, new_step_size_max, energy_change)`.
fn handle_nans(
    prev: &MclmcState,
    next: MclmcState,
    step_size: f32,
    step_size_max: f32,
    energy_change: f32,
) -> (bool, MclmcState, f32, f32) {
    let all_finite = next.position.iter().all(|x| x.is_finite())
        && next.logdensity.is_finite();

    if all_finite {
        (true, next, step_size_max, energy_change)
    } else {
        let reduced = (step_size * 0.8).min(step_size_max);
        (false, prev.clone(), reduced, 0.0)
    }
}
