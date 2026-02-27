//! Post-sampling inference: metrics, diagnostics, and reporting.
//!
//! Mirrors `src/inference/` from the Python reference.

pub mod diagnostics;

pub use diagnostics::{effective_sample_size, potential_scale_reduction};

use crate::sampling::ensemble::ChainResult;

// ── Diagnostics ───────────────────────────────────────────────────────────────

/// Compute ESS and R-hat across chains for each parameter dimension.
///
/// `results`: one `ChainResult` per chain, all with the same parameter dim.
pub fn diagnostics(results: &[ChainResult]) -> Diagnostics {
    let n_chains = results.len();
    if n_chains == 0 {
        return Diagnostics::default();
    }
    let dim = results[0].samples.first().map(|s| s.len()).unwrap_or(0);
    let n_draws = results[0].samples.len();

    let mut ess_per_dim: Vec<f64> = Vec::with_capacity(dim);
    let mut rhat_per_dim: Vec<f64> = Vec::with_capacity(dim);

    for d in 0..dim {
        let chains: Vec<Vec<f64>> = results
            .iter()
            .map(|r| r.samples.iter().map(|s| f64::from(s[d])).collect())
            .collect();

        // ESS: sum across chains for this dimension.
        let ess: f64 = chains
            .iter()
            .map(|c| effective_sample_size(c))
            .sum::<f64>();
        ess_per_dim.push(ess);

        let chain_refs: Vec<&[f64]> = chains.iter().map(|c| c.as_slice()).collect();
        let rhat = potential_scale_reduction(&chain_refs);
        rhat_per_dim.push(rhat);
    }

    let mean_ess = if dim > 0 {
        ess_per_dim.iter().sum::<f64>() / dim as f64
    } else {
        0.0
    };
    let max_rhat = rhat_per_dim
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    Diagnostics {
        n_chains,
        n_draws,
        dim,
        mean_ess,
        max_rhat,
        ess_per_dim,
        rhat_per_dim,
    }
}

/// Summary diagnostics for a completed sampling run.
#[derive(Debug, Default)]
pub struct Diagnostics {
    pub n_chains: usize,
    pub n_draws: usize,
    pub dim: usize,
    pub mean_ess: f64,
    pub max_rhat: f64,
    pub ess_per_dim: Vec<f64>,
    pub rhat_per_dim: Vec<f64>,
}

impl std::fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Diagnostics {{ chains={}, draws={}, dim={}, mean_ESS={:.1}, max_Rhat={:.4} }}",
            self.n_chains, self.n_draws, self.dim, self.mean_ess, self.max_rhat
        )
    }
}

// ── Prediction ────────────────────────────────────────────────────────────────

/// Compute ensemble predictive mean and variance over posterior samples.
///
/// `predictions`: `[n_samples, batch, out_dim]`.
/// Returns `(mean, variance)` each of shape `[batch, out_dim]`.
pub fn ensemble_stats(predictions: &[Vec<Vec<f32>>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = predictions.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let batch = predictions[0].len();
    let out_dim = predictions[0].first().map(|r| r.len()).unwrap_or(0);

    let mut mean = vec![vec![0.0f32; out_dim]; batch];
    let mut var = vec![vec![0.0f32; out_dim]; batch];

    for pred in predictions {
        for (b, row) in pred.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                mean[b][d] += v / n as f32;
            }
        }
    }
    for pred in predictions {
        for (b, row) in pred.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                let diff = v - mean[b][d];
                var[b][d] += diff * diff / n as f32;
            }
        }
    }

    (mean, var)
}
