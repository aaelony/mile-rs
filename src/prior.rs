//! Prior distributions over network parameters.
//!
//! Mirrors `src/training/priors.py`.  All priors operate on a **flat**
//! parameter tensor of shape `[D]` and return a scalar log-probability.
//!
//! The prior must be differentiable (used inside `value_and_grad`), so all
//! operations are expressed as Burn tensor ops so that the autodiff graph
//! is preserved end-to-end.

use burn::tensor::{backend::Backend, Tensor};

use crate::config::PriorConfig;

// ── Prior enum ────────────────────────────────────────────────────────────────

/// A prior distribution over the flat parameter vector.
#[derive(Debug, Clone)]
pub enum Prior {
    /// N(0, 1) on every parameter.
    StandardNormal,
    /// N(loc, scale) on every parameter.
    Normal { loc: f32, scale: f32 },
    /// Laplace(loc, scale) on every parameter.
    Laplace { loc: f32, scale: f32 },
}

impl Prior {
    pub fn from_config(cfg: &PriorConfig) -> Self {
        match cfg {
            PriorConfig::StandardNormal => Self::StandardNormal,
            PriorConfig::Normal { loc, scale } => Self::Normal {
                loc: *loc,
                scale: *scale,
            },
            PriorConfig::Laplace { loc, scale } => Self::Laplace {
                loc: *loc,
                scale: *scale,
            },
        }
    }

    /// Compute `sum log p(w)` for every element of `flat_params`.
    ///
    /// The input tensor **must** have `require_grad()` set if this is called
    /// inside `value_and_grad`; the grad flows through the arithmetic here.
    pub fn log_prior<B: Backend>(&self, flat_params: Tensor<B, 1>) -> Tensor<B, 1> {
        match self {
            Self::StandardNormal => log_prior_normal(flat_params, 0.0, 1.0),
            Self::Normal { loc, scale } => log_prior_normal(flat_params, *loc, *scale),
            Self::Laplace { loc, scale } => log_prior_laplace(flat_params, *loc, *scale),
        }
    }
}

// ── Log-prior implementations ─────────────────────────────────────────────────

/// `sum_i log N(w_i | loc, scale)`
///
/// = -D/2 * log(2π) - D * log(scale) - sum_i (w_i - loc)^2 / (2 * scale^2)
///
/// The constant term is omitted because it does not affect the gradient.
fn log_prior_normal<B: Backend>(params: Tensor<B, 1>, loc: f32, scale: f32) -> Tensor<B, 1> {
    let device = params.device();
    let diff = params - Tensor::full([1], loc, &device);
    let var = scale * scale;
    // -0.5 * sum((w - loc)^2 / scale^2)
    let sum_sq = (diff.powi_scalar(2) / var).sum();
    sum_sq * -0.5
}

/// `sum_i log Laplace(w_i | loc, scale)`
///
/// = -D * log(2 * scale) - sum_i |w_i - loc| / scale
///
/// Constant term omitted as above.
fn log_prior_laplace<B: Backend>(params: Tensor<B, 1>, loc: f32, scale: f32) -> Tensor<B, 1> {
    let device = params.device();
    let diff = params - Tensor::full([1], loc, &device);
    let sum_abs = diff.abs().sum();
    sum_abs / (-scale)
}
