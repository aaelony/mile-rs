//! mile-rs — Microcanonical Langevin Ensembles for Bayesian Neural Networks.
//!
//! A pure-Rust implementation of MILE (arxiv 2502.06335) using:
//! - Burn for neural networks, autodiff, and warmstart training
//! - A hand-ported MCLMC sampler (no existing Rust crate provides this)
//! - nuts-rs for the NUTS comparison baseline
//! - Rayon for parallel chain execution (CPU backend only)

pub mod config;
pub mod dataset;
pub mod inference;
pub mod models;
pub mod prior;
pub mod probabilistic;
pub mod sampling;
pub mod training;

// ── Backend type aliases ──────────────────────────────────────────────────────
//
// The backend is selected at runtime from the TOML config:
//   backend = "cpu"  →  CpuBackend = Autodiff<NdArray<f32>>   (parallel via Rayon)
//   backend = "gpu"  →  GpuBackend = Autodiff<Wgpu>           (sequential, GPU dispatch)
//
// MCMC state lives in flat Vec<f32>; Burn tensors are only used inside
// BnnLogPosterior::value_and_grad for the autodiff forward/backward pass.

use burn::backend::{
    ndarray::NdArrayDevice,
    wgpu::WgpuDevice,
    Autodiff, NdArray, Wgpu,
};

/// CPU backend: pure f32 NdArray with autodiff.  Chains run in parallel via Rayon.
pub type CpuBackend = Autodiff<NdArray<f32>>;

/// GPU backend: WebGPU with autodiff.  Chains run sequentially to avoid
/// contending on the single GPU command queue.
pub type GpuBackend = Autodiff<Wgpu>;

/// Default device for the CPU backend.
pub fn cpu_device() -> NdArrayDevice {
    NdArrayDevice::default()
}

/// Default device for the GPU (Wgpu) backend.
pub fn gpu_device() -> WgpuDevice {
    WgpuDevice::default()
}

// ── Common error type ─────────────────────────────────────────────────────────

/// Top-level error type for the MILE pipeline.
#[derive(Debug, thiserror::Error)]
pub enum MileError {
    #[error("Logp evaluation failed: {0}")]
    Logp(#[from] probabilistic::LogpError),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}
