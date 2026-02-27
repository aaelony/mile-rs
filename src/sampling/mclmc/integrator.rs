//! Isokinetic leapfrog integrator for MCLMC.
//!
//! Implements the velocity-Verlet (leapfrog) scheme with the isokinetic
//! constraint `‖p‖ = 1`.  The constraint is enforced after each momentum
//! half-step by projecting `p` back to the unit sphere.
//!
//! The McLachlan 4th-order integrator used in the Python blackjax implementation
//! is a composition of these half/full steps; it can be layered on top later.
//!
//! Reference: Robnik et al., "Microcanonical Hamiltonian Monte Carlo" (2023).

use crate::{
    probabilistic::LogpError,
    sampling::mclmc::state::{axpy, normalize_inplace},
};

/// Output of a single integrator step.
pub struct IntegratorResult {
    pub position: Vec<f32>,
    pub momentum: Vec<f32>,
    pub logdensity: f32,
    pub logdensity_grad: Vec<f32>,
    /// Change in the isokinetic Hamiltonian `H = -logp + ½‖p‖²`.
    /// Because `‖p‖ = 1` is conserved by construction, this equals
    /// `logp_old - logp_new` plus any integrator error.
    pub energy_change: f32,
}

/// Perform one velocity-Verlet step with isokinetic projection.
///
/// # Arguments
///
/// * `position` / `momentum` / `logdensity_grad` — current state.
/// * `step_size` — leapfrog step size `ε`.
/// * `sqrt_diag_cov` — preconditioning; use all-ones for no preconditioning.
/// * `logp_fn` — evaluates `(log p(q), ∇ log p(q))` for a given position.
/// * `logdensity_old` — `log p` at the current position (avoids a redundant eval).
pub fn leapfrog_step(
    position: &[f32],
    momentum: &[f32],
    logdensity_grad: &[f32],
    logdensity_old: f32,
    step_size: f32,
    sqrt_diag_cov: &[f32],
    logp_fn: &dyn Fn(&[f32]) -> Result<(f32, Vec<f32>), LogpError>,
) -> Result<IntegratorResult, LogpError> {
    // ── Half momentum step: p_{1/2} = p + (ε/2) * ∇logp(q) ─────────────────
    let mut p_half = momentum.to_vec();
    axpy(&mut p_half, step_size * 0.5, logdensity_grad);

    // Apply preconditioning: scale by sqrt_diag_cov element-wise.
    // When sqrt_diag_cov = 1 everywhere this is a no-op.
    apply_preconditioner(&mut p_half, sqrt_diag_cov);

    // Enforce isokinetic constraint.
    normalize_inplace(&mut p_half);

    // ── Full position step: q_new = q + ε * p_{1/2} ─────────────────────────
    let mut q_new = position.to_vec();
    axpy(&mut q_new, step_size, &p_half);

    // ── Evaluate logp and grad at new position ───────────────────────────────
    let (logdensity_new, grad_new) = logp_fn(&q_new)?;

    // ── Half momentum step: p_new = p_{1/2} + (ε/2) * ∇logp(q_new) ─────────
    let mut p_new = p_half;
    axpy(&mut p_new, step_size * 0.5, &grad_new);
    apply_preconditioner(&mut p_new, sqrt_diag_cov);
    normalize_inplace(&mut p_new);

    // ── Energy change (isokinetic Hamiltonian drift) ─────────────────────────
    // H_old = -logp_old + 0.5  (‖p‖² = 1)
    // H_new = -logp_new + 0.5
    // ΔH = logp_old - logp_new
    let energy_change = logdensity_old - logdensity_new;

    Ok(IntegratorResult {
        position: q_new,
        momentum: p_new,
        logdensity: logdensity_new,
        logdensity_grad: grad_new,
        energy_change,
    })
}

/// Apply diagonal preconditioning: `p[i] *= sqrt_diag_cov[i]`.
///
/// This rescales the momentum in the whitened coordinate system.
/// When all entries of `sqrt_diag_cov` are 1.0 this is a no-op.
fn apply_preconditioner(p: &mut [f32], sqrt_diag_cov: &[f32]) {
    for (pi, &s) in p.iter_mut().zip(sqrt_diag_cov) {
        *pi *= s;
    }
}
