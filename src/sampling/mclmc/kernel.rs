//! MCLMC kernel: one full Markov transition.
//!
//! A single MCLMC step consists of:
//!  1. One leapfrog integrator step (position + deterministic momentum update).
//!  2. Partial momentum refresh (stochastic, keeps chain ergodic).
//!
//! The partial refresh formula is:
//!   p_new = √(1 - α²) · p + α · ξ/‖ξ‖
//! where `ξ ~ N(0, I)` and `α = ε / L`.
//! This mixes the deterministic momentum with fresh noise while preserving
//! `‖p_new‖ = 1` in expectation.
//!
//! Reference: Robnik et al. (2023), Algorithm 1.

use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

use crate::{
    probabilistic::LogpError,
    sampling::mclmc::{
        integrator::leapfrog_step,
        state::{axpy, normalize_inplace, MclmcParams, MclmcState},
    },
};

/// Perform one MCLMC kernel step.
///
/// Returns the next `MclmcState` and the energy change from the integrator,
/// which is used during warmup to adapt `step_size`.
pub fn mclmc_step(
    state: &MclmcState,
    params: &MclmcParams,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
) -> Result<(MclmcState, f32), LogpError> {
    let dim = state.position.len();

    // ── 1. Leapfrog step ──────────────────────────────────────────────────────
    let result = leapfrog_step(
        &state.position,
        &state.momentum,
        &state.logdensity_grad,
        state.logdensity,
        params.step_size,
        &params.sqrt_diag_cov,
        logp_fn,
    )?;

    // ── 2. Partial momentum refresh ───────────────────────────────────────────
    let alpha = (params.step_size / params.l).min(1.0);
    let cos_alpha = (1.0 - alpha * alpha).sqrt();

    // Draw unit Gaussian noise and normalise to the unit sphere.
    let mut noise: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
    normalize_inplace(&mut noise);

    // Mix: p_refreshed = cos_alpha * p_new + alpha * noise
    let mut p_refreshed: Vec<f32> = result.momentum.iter().map(|&p| cos_alpha * p).collect();
    axpy(&mut p_refreshed, alpha, &noise);

    // Renormalise to exactly unit length (corrects floating-point drift).
    normalize_inplace(&mut p_refreshed);

    let next_state = MclmcState {
        position: result.position,
        momentum: p_refreshed,
        logdensity: result.logdensity,
        logdensity_grad: result.logdensity_grad,
    };

    Ok((next_state, result.energy_change))
}

/// Initialise a chain from a starting position.
///
/// Draws a random unit-sphere momentum and evaluates logp + grad.
pub fn init_chain(
    position: Vec<f32>,
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
    rng: &mut impl Rng,
) -> Result<MclmcState, LogpError> {
    let dim = position.len();

    // Random momentum on the unit sphere
    let mut momentum: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
    normalize_inplace(&mut momentum);

    let (logdensity, logdensity_grad) = logp_fn(&position)?;

    Ok(MclmcState {
        position,
        momentum,
        logdensity,
        logdensity_grad,
    })
}
