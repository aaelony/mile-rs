//! MCLMC sampler state and hyperparameters.

/// The current state of one MCLMC chain.
///
/// All vectors have length `dim` (the flat parameter-space dimension).
///
/// The isokinetic constraint keeps `‖momentum‖ = 1.0` at all times;
/// numerical drift is corrected after every integrator step.
#[derive(Debug, Clone)]
pub struct MclmcState {
    /// Current position in parameter space (flat, dim = D).
    pub position: Vec<f32>,
    /// Current momentum, constrained to the unit sphere: `‖momentum‖ = 1`.
    pub momentum: Vec<f32>,
    /// `log p(position)` at the current point.
    pub logdensity: f32,
    /// Gradient of `log p` at the current position, shape `[D]`.
    pub logdensity_grad: Vec<f32>,
}

impl MclmcState {
    /// Construct an initial state given a position and a logp+grad evaluation.
    pub fn new(
        position: Vec<f32>,
        momentum: Vec<f32>,
        logdensity: f32,
        logdensity_grad: Vec<f32>,
    ) -> Self {
        debug_assert!(
            (norm(&momentum) - 1.0).abs() < 1e-4,
            "Initial momentum must be unit-normalised"
        );
        Self {
            position,
            momentum,
            logdensity,
            logdensity_grad,
        }
    }
}

/// Tunable hyperparameters for MCLMC.
///
/// Produced by `warmup::find_l_and_step_size` and then held fixed during
/// the main sampling phase.
#[derive(Debug, Clone)]
pub struct MclmcParams {
    /// Leapfrog step size `ε`.
    pub step_size: f32,
    /// Momentum decoherence length `L`.  Controls how quickly the momentum is
    /// refreshed: the refresh fraction per step is `ε / L`.
    pub l: f32,
    /// Square-root of the diagonal preconditioning matrix.
    /// Shape `[D]`, all-ones when `diagonal_preconditioning = false`.
    pub sqrt_diag_cov: Vec<f32>,
}

impl MclmcParams {
    pub fn initial(dim: usize, step_size_init: f32) -> Self {
        let l_init = (dim as f32).sqrt().max(15.0);
        Self {
            step_size: step_size_init,
            l: l_init,
            sqrt_diag_cov: vec![1.0; dim],
        }
    }
}

// ── Small vector helpers ──────────────────────────────────────────────────────

/// Euclidean norm of a slice.
pub(crate) fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalise a vector in-place to unit length.
pub(crate) fn normalize_inplace(v: &mut [f32]) {
    let n = norm(v);
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Element-wise `a += scale * b`.
pub(crate) fn axpy(a: &mut [f32], scale: f32, b: &[f32]) {
    for (ai, bi) in a.iter_mut().zip(b) {
        *ai += scale * bi;
    }
}
