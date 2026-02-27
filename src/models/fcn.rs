//! Fully-connected network (FCN) in two forms.
//!
//! - `FcnModule<B>` — Burn `Module` used for warmstart training.
//! - `fcn_forward(params, x)` — pure function used during MCMC sampling, where
//!   weights arrive as a flat position tensor so autodiff can flow through them.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

use crate::{config::FcnConfig, models::params::FcnLayerParams};

// ── Module form (warmstart training) ─────────────────────────────────────────

/// A fully-connected network as a Burn `Module`.
///
/// Architecture mirrors the Python `FullyConnected` building block:
/// `Linear → ReLU → ... → Linear` (no activation on the last layer).
#[derive(Module, Debug)]
pub struct FcnModule<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}

/// Re-export as `Fcn` for brevity at call sites.
pub type Fcn<B> = FcnModule<B>;

impl<B: Backend> FcnModule<B> {
    /// Forward pass: `x` has shape `[batch, in_dim]`, output `[batch, out_dim]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = self.layers.len();
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            if i < n - 1 {
                h = self.activation.forward(h);
            }
        }
        h
    }

    /// Extract all parameters as a flat `Vec<f32>`, in the same order as
    /// `FcnParamFlattener`:  weight (row-major) then bias, layer by layer.
    pub fn to_flat_vec(&self) -> Vec<f32> {
        let mut flat: Vec<f32> = Vec::new();
        for layer in &self.layers {
            // weight: Param<Tensor<B, 2>>, shape [out, in]
            let w: Vec<f32> = layer
                .weight
                .val()
                .into_data()
                .to_vec()
                .expect("weight to_vec failed");
            flat.extend(w);

            // bias: Option<Param<Tensor<B, 1>>>
            if let Some(ref bias) = layer.bias {
                let b: Vec<f32> = bias
                    .val()
                    .into_data()
                    .to_vec()
                    .expect("bias to_vec failed");
                flat.extend(b);
            }
        }
        flat
    }
}

/// Build an `FcnModule` from an `FcnConfig`.
pub fn build_fcn<B: Backend>(cfg: &FcnConfig, device: &B::Device) -> FcnModule<B> {
    let layers: Vec<Linear<B>> = cfg
        .layer_shapes()
        .iter()
        .map(|&(in_dim, out_dim)| {
            LinearConfig::new(in_dim, out_dim)
                .with_bias(cfg.use_bias)
                .init(device)
        })
        .collect();

    FcnModule {
        layers,
        activation: Relu::new(),
    }
}

// ── Pure-function form (MCMC sampling) ───────────────────────────────────────

/// Stateless FCN forward pass.
///
/// `params` is a `Vec<FcnLayerParams<B>>` produced by `FcnParamFlattener::unflatten`.
/// Because every tensor operation is a Burn op, the autodiff graph flows through
/// this function when `params` was built from a `require_grad()` position tensor.
///
/// Activation: ReLU on every layer except the last.
pub fn fcn_forward<B: Backend>(
    params: &[FcnLayerParams<B>],
    x: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let n = params.len();
    let mut h = x;
    for (i, lp) in params.iter().enumerate() {
        // h = h @ W^T + b
        h = h.matmul(lp.weight.clone().transpose());
        if let Some(ref bias) = lp.bias {
            // broadcast bias across batch dimension
            h = h + bias.clone().unsqueeze_dim(0);
        }
        if i < n - 1 {
            h = burn::tensor::activation::relu(h);
        }
    }
    h
}
