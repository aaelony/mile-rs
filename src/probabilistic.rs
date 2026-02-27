//! Probabilistic model: `log p(θ | data) ∝ log p(θ) + n_batches · log p(data | θ)`.
//!
//! Mirrors `src/training/probabilistic.py::ProbabilisticModel`.
//!
//! `BnnLogPosterior<B>` implements the `LogPosterior` trait, which is the
//! central seam consumed by both the MCLMC sampler and the nuts-rs bridge.

use burn::{
    tensor::{
        backend::{AutodiffBackend, Backend},
        Shape, Tensor, TensorData,
    },
};
use thiserror::Error;

use crate::{
    config::{FcnConfig, PriorConfig, Task},
    models::{
        fcn::fcn_forward,
        params::FcnParamFlattener,
    },
    prior::Prior,
};

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum LogpError {
    #[error("Non-recoverable logp error: {0}")]
    Fatal(String),
    #[error("Recoverable logp error: {0}")]
    Recoverable(String),
}

// Implement nuts-rs's LogpError trait so NutsBridge can forward this error.
impl nuts_rs::LogpError for LogpError {
    fn is_recoverable(&self) -> bool {
        matches!(self, LogpError::Recoverable(_))
    }
}

// ── Central trait ─────────────────────────────────────────────────────────────

/// Computes log unnormalised posterior **and** its gradient w.r.t. the flat
/// parameter vector in a single autodiff pass.
///
/// Implementors are `Send + Sync` so they can be cloned per chain in Rayon.
pub trait LogPosterior: Send + Sync {
    /// Dimension of the flat parameter space.
    fn dim(&self) -> usize;

    /// Returns `(log_posterior, gradient_vector)`.
    ///
    /// `flat_pos`: current MCMC position as a slice of `f32`.
    fn value_and_grad(&self, flat_pos: &[f32]) -> Result<(f32, Vec<f32>), LogpError>;
}

// ── BNN log-posterior ─────────────────────────────────────────────────────────

/// Concrete implementation for an FCN BNN with full-batch data.
///
/// Uses Burn autodiff internally:
/// 1. Convert `flat_pos: &[f32]` → `Tensor<B, 1>` with `require_grad()`.
/// 2. Unflatten → per-layer tensors.
/// 3. Run `fcn_forward` (all Burn ops → graph is built).
/// 4. Compute log-likelihood + log-prior.
/// 5. `backward()` → extract gradient for the position tensor.
pub struct BnnLogPosterior<B: AutodiffBackend> {
    pub flattener: FcnParamFlattener,
    pub prior: Prior,
    pub task: Task,
    /// n_batches > 1 for mini-batch; 1 for full-batch (current implementation).
    pub n_batches: f32,
    /// Training features, shape `[N, in_dim]`, on the **inner** (non-autodiff) backend.
    pub train_x: Tensor<B::InnerBackend, 2>,
    /// Training labels.
    ///   - Regression: shape `[N]` (f32 targets)
    ///   - Classification: shape `[N]` (integer class indices stored as f32)
    pub train_y: Tensor<B::InnerBackend, 1>,
    pub device: B::Device,
}

// SAFETY: Burn's NdArray tensors are Send + Sync for f32.
unsafe impl<B: AutodiffBackend> Send for BnnLogPosterior<B>
where
    B::InnerBackend: Backend,
    Tensor<B::InnerBackend, 2>: Send,
    Tensor<B::InnerBackend, 1>: Send,
{
}
unsafe impl<B: AutodiffBackend> Sync for BnnLogPosterior<B>
where
    B::InnerBackend: Backend,
    Tensor<B::InnerBackend, 2>: Sync,
    Tensor<B::InnerBackend, 1>: Sync,
{
}

impl<B: AutodiffBackend> BnnLogPosterior<B> {
    pub fn new(
        fcn_cfg: &FcnConfig,
        prior_cfg: &PriorConfig,
        task: Task,
        train_x: Tensor<B::InnerBackend, 2>,
        train_y: Tensor<B::InnerBackend, 1>,
        device: B::Device,
    ) -> Self {
        Self {
            flattener: FcnParamFlattener::from_config(fcn_cfg),
            prior: Prior::from_config(prior_cfg),
            task,
            n_batches: 1.0,
            train_x,
            train_y,
            device,
        }
    }

    // ── Log-likelihood ────────────────────────────────────────────────────────

    /// `log p(y | θ, x)` for the current task.
    fn log_likelihood(
        &self,
        logits: Tensor<B, 2>,
        y: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        match self.task {
            Task::Regression => self.log_likelihood_regression(logits, y),
            Task::Classification { .. } => self.log_likelihood_classification(logits, y),
        }
    }

    /// Gaussian NLL: output logits are `[batch, 2]` = (mu, log_sigma).
    fn log_likelihood_regression(
        &self,
        logits: Tensor<B, 2>,
        y: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let [batch, _] = logits.dims();
        let mu = logits.clone().slice([0..batch, 0..1]).squeeze::<1>();
        let log_sigma = logits.slice([0..batch, 1..2]).squeeze::<1>();
        let sigma = log_sigma.exp().clamp(1e-6, 1e6);
        let diff = y - mu;
        // log N(y | mu, sigma) = -0.5*(y-mu)^2/sigma^2 - log(sigma) - 0.5*log(2pi)
        let log_lik = -(diff.clone().powi_scalar(2) / sigma.clone().powi_scalar(2)) * 0.5
            - sigma.log()
            - (2.0 * std::f32::consts::PI).ln() * 0.5;
        log_lik.sum().unsqueeze_dim(0)
    }

    /// Categorical cross-entropy: `logits` shape `[batch, n_classes]`.
    fn log_likelihood_classification(
        &self,
        logits: Tensor<B, 2>,
        y: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let [_batch, n_classes] = logits.dims();
        // Capture device before any moves.
        let device = logits.device();
        // log_pmf = logits - logsumexp(logits, dim=-1, keepdim=True)
        // sum_dim(1) in Burn keeps the dimension, so the result is already [batch, 1].
        let lse = logits.clone().exp().sum_dim(1).log(); // [batch, 1]
        let log_pmf = logits.clone() - lse; // [batch, n_classes] broadcasts against [batch, 1]

        // Gather log_pmf at the true class indices
        // y: [batch] of integer indices stored as f32 → cast to int for gather
        let y_int = y.clone().into_data();
        let indices: Vec<usize> = y_int
            .to_vec::<f32>()
            .expect("label to_vec failed")
            .iter()
            .map(|&v| v as usize)
            .collect();
        let mut log_lik_val = 0.0f32;
        let log_pmf_data: Vec<f32> = log_pmf.into_data().to_vec().expect("logpmf to_vec failed");
        for (b, &cls) in indices.iter().enumerate() {
            log_lik_val += log_pmf_data[b * n_classes + cls];
        }
        let data = TensorData::new(vec![log_lik_val], Shape::new([1]));
        Tensor::from_data(data, &device)
    }
}

impl<B: AutodiffBackend> LogPosterior for BnnLogPosterior<B> {
    fn dim(&self) -> usize {
        self.flattener.param_dim()
    }

    fn value_and_grad(&self, flat_pos: &[f32]) -> Result<(f32, Vec<f32>), LogpError> {
        let device = &self.device;

        // ── 1. Build differentiable position tensor ───────────────────────────
        let pos_data = TensorData::new(flat_pos.to_vec(), Shape::new([flat_pos.len()]));
        let pos: Tensor<B, 1> = Tensor::from_data(pos_data, device).require_grad();

        // ── 2. Unflatten into per-layer weight/bias tensors ───────────────────
        let layer_params = self.flattener.unflatten(pos.clone(), device);

        // ── 3. Forward pass (pure function, no Module state) ──────────────────
        // Bring data tensors onto the autodiff backend.
        let x: Tensor<B, 2> = Tensor::from_inner(self.train_x.clone());
        let logits = fcn_forward(&layer_params, x);

        // ── 4. Log-likelihood ─────────────────────────────────────────────────
        let y: Tensor<B, 1> = Tensor::from_inner(self.train_y.clone());
        let log_lik = self.log_likelihood(logits, y);

        // ── 5. Log-prior ──────────────────────────────────────────────────────
        let log_prior = self.prior.log_prior(pos.clone());

        // ── 6. Unnormalised posterior ─────────────────────────────────────────
        let log_post = log_prior + log_lik * self.n_batches;

        // ── 7. Scalar value (clone before backward consumes the tensor) ───────
        // into_scalar() returns B::FloatElem (associated type); use into_data()
        // to extract the concrete f32 value regardless of backend.
        let lp_val: f32 = log_post
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(f32::NEG_INFINITY);

        if !lp_val.is_finite() {
            return Err(LogpError::Recoverable(format!(
                "Non-finite log-posterior: {lp_val}"
            )));
        }

        // ── 8. Backward pass ──────────────────────────────────────────────────
        let mut grads = log_post.backward();

        // ── 9. Extract gradient w.r.t. the position tensor ───────────────────
        let grad_tensor = pos
            .grad_remove(&mut grads)
            .ok_or_else(|| LogpError::Fatal("No gradient for position tensor".into()))?;

        let grad_vec: Vec<f32> = grad_tensor
            .into_data()
            .to_vec()
            .map_err(|e| LogpError::Fatal(format!("Gradient extraction failed: {e:?}")))?;

        Ok((lp_val, grad_vec))
    }
}
