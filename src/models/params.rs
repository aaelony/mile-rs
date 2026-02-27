//! Parameter flattening / unflattening for the FCN.
//!
//! # The flattening bridge
//!
//! JAX uses `ravel_pytree` to flatten nested parameter dicts to a 1-D array
//! for MCMC.  Here we replicate that with explicit offset arithmetic.
//!
//! Layout for an FCN with layer shapes `[(in0, out0), (in1, out1), ...]`:
//!
//! ```text
//! [  weight_0 (out0 × in0, row-major)  |  bias_0 (out0)  |
//!    weight_1 (out1 × out0, row-major) |  bias_1 (out1)  | ... ]
//! ```
//!
//! This must match the order used in `Fcn::to_flat_vec` (extraction from the
//! trained Module) so that warmstart → MCMC initialisation is correct.

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

use crate::config::FcnConfig;

// ── Layer parameter holder ────────────────────────────────────────────────────

/// Weight and bias tensors for a single linear layer.
///
/// `weight` has shape `[out_dim, in_dim]` (matches Burn's `Linear` convention).
/// `bias`   has shape `[out_dim]`.
pub struct FcnLayerParams<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
}

// ── Flattener ─────────────────────────────────────────────────────────────────

/// Knows the shape of every layer and can convert between the flat position
/// vector (used by MCMC) and per-layer tensors (used by the forward pass).
#[derive(Debug, Clone)]
pub struct FcnParamFlattener {
    /// `(in_dim, out_dim)` for each layer, in order.
    pub layer_shapes: Vec<(usize, usize)>,
    pub use_bias: bool,
}

impl FcnParamFlattener {
    pub fn from_config(cfg: &FcnConfig) -> Self {
        Self {
            layer_shapes: cfg.layer_shapes(),
            use_bias: cfg.use_bias,
        }
    }

    /// Total number of scalar parameters.
    pub fn param_dim(&self) -> usize {
        self.layer_shapes.iter().map(|&(i, o)| {
            let w = i * o;
            let b = if self.use_bias { o } else { 0 };
            w + b
        }).sum()
    }

    /// Slice offsets for every weight and bias tensor (start, length).
    fn offsets(&self) -> Vec<(usize, usize, Option<(usize, usize)>)> {
        let mut out = Vec::new();
        let mut offset = 0;
        for &(in_dim, out_dim) in &self.layer_shapes {
            let w_start = offset;
            let w_len = in_dim * out_dim;
            offset += w_len;

            let bias_range = if self.use_bias {
                let b_start = offset;
                offset += out_dim;
                Some((b_start, out_dim))
            } else {
                None
            };

            out.push((w_start, w_len, bias_range));
        }
        out
    }

    /// Flat `Tensor<B, 1>` → per-layer `FcnLayerParams`.
    ///
    /// All operations are Burn tensor ops so the autodiff graph is preserved
    /// when `flat` has `require_grad()` set.
    pub fn unflatten<B: Backend>(
        &self,
        flat: Tensor<B, 1>,
        _device: &B::Device,
    ) -> Vec<FcnLayerParams<B>> {
        let offsets = self.offsets();
        offsets
            .iter()
            .zip(self.layer_shapes.iter())
            .map(|(&(w_start, w_len, bias_range), &(in_dim, out_dim))| {
                let weight = flat
                    .clone()
                    .narrow(0, w_start, w_len)
                    .reshape([out_dim, in_dim]);

                let bias = bias_range.map(|(b_start, b_len)| {
                    flat.clone().narrow(0, b_start, b_len)
                });

                FcnLayerParams { weight, bias }
            })
            .collect()
    }

    /// Extract flat `Vec<f32>` from raw slices of weights and biases.
    ///
    /// Used after warmstart training to convert the trained Module parameters
    /// into the initial MCMC position for each chain.
    ///
    /// `layer_data`: `(weight_flat, bias_flat_or_empty)` per layer, in order.
    pub fn flatten_raw(&self, layer_data: &[(Vec<f32>, Vec<f32>)]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_dim());
        for (w, b) in layer_data {
            out.extend_from_slice(w);
            if self.use_bias {
                out.extend_from_slice(b);
            }
        }
        out
    }

    /// Build a `Tensor<B, 1>` from a flat `&[f32]` slice.
    ///
    /// Convenience wrapper used when creating the initial MCMC position tensor.
    pub fn to_tensor<B: Backend>(&self, flat: &[f32], device: &B::Device) -> Tensor<B, 1> {
        let data = TensorData::new(flat.to_vec(), Shape::new([flat.len()]));
        Tensor::from_data(data, device)
    }
}
