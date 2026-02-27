//! NUTS bridge: connects the `LogPosterior` trait to `nuts-rs`.
//!
//! `nuts-rs` requires the caller to supply gradients manually through the
//! `CpuLogpFunc` trait.  `NutsBridge` wraps any `LogPosterior` implementor
//! and forwards the `value_and_grad` result, converting between f32 (Burn /
//! MCLMC) and f64 (nuts-rs convention).

use std::{collections::HashMap, sync::Arc};

use nuts_rs::{Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, HasDims, Settings};

use crate::probabilistic::{LogpError, LogPosterior};

// ── NutsBridge ────────────────────────────────────────────────────────────────

/// Wraps a `LogPosterior` to implement nuts-rs's `CpuLogpFunc`.
pub struct NutsBridge<M: LogPosterior> {
    model: Arc<M>,
}

impl<M: LogPosterior> NutsBridge<M> {
    pub fn new(model: Arc<M>) -> Self {
        Self { model }
    }
}

impl<M: LogPosterior> HasDims for NutsBridge<M> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("unconstrained_parameter".to_string(), self.model.dim() as u64),
        ])
    }
}

impl<M: LogPosterior + 'static> CpuLogpFunc for NutsBridge<M> {
    type LogpError = LogpError;
    type ExpandedVector = Vec<f64>;
    type FlowParameters = ();

    fn dim(&self) -> usize {
        self.model.dim()
    }

    /// `position` and `grad` are `f64` slices (nuts-rs convention).
    /// We convert to `f32`, evaluate via `LogPosterior`, and convert back.
    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        // f64 → f32
        let pos_f32: Vec<f32> = position.iter().map(|&x| x as f32).collect();

        let (lp_f32, grad_f32) = self.model.value_and_grad(&pos_f32)?;

        // f32 → f64
        for (g_out, &g_val) in grad.iter_mut().zip(grad_f32.iter()) {
            *g_out = f64::from(g_val);
        }

        Ok(f64::from(lp_f32))
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(position.to_vec())
    }
}

// ── NUTS sampling for a single chain ─────────────────────────────────────────

/// Draw `n_samples` from one NUTS chain, starting at `init_pos`.
///
/// `warmup_steps` are discarded.  Returns a `Vec` of flat parameter vectors.
pub fn nuts_sample_chain<M>(
    model: Arc<M>,
    sampler_cfg: &crate::config::SamplerConfig,
    chain_id: usize,
    init_pos: &[f32],
    pb: indicatif::ProgressBar,
) -> Result<Vec<Vec<f32>>, crate::MileError>
where
    M: LogPosterior + 'static,
{
    let bridge = NutsBridge::new(model);
    let math = CpuMath::new(bridge);

    let mut settings = DiagGradNutsSettings::default();
    settings.num_tune = sampler_cfg.warmup_steps as u64;
    settings.maxdepth = sampler_cfg.nuts_max_depth;

    // Use nuts_rs::rand (rand 0.10) for the RNG so trait bounds are satisfied.
    let mut rng = nuts_rs::rand::rng();
    let mut chain = settings.new_chain(chain_id as u64, math, &mut rng);

    let init_f64: Vec<f64> = init_pos.iter().map(|&x| f64::from(x)).collect();
    chain
        .set_position(&init_f64)
        .map_err(|e| crate::MileError::Config(format!("NUTS set_position failed: {e:?}")))?;

    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(sampler_cfg.n_samples);
    let mut step = 0usize;

    pb.set_message("warmup");

    for _ in 0..sampler_cfg.n_samples * sampler_cfg.n_thinning {
        let (draw, _info): (Box<[f64]>, _) = chain
            .draw()
            .map_err(|e| crate::MileError::Config(format!("NUTS draw failed: {e:?}")))?;

        step += 1;

        // nuts-rs internally handles warmup for the first `num_tune` draws.
        if step == sampler_cfg.warmup_steps + 1 {
            pb.set_message("sampling");
        }

        if step % sampler_cfg.n_thinning == 0 {
            let pos_f32: Vec<f32> = draw.iter().map(|&x| x as f32).collect();
            samples.push(pos_f32);
        }

        pb.inc(1);
    }

    Ok(samples)
}
