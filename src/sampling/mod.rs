//! MCMC samplers.
//!
//! - `mclmc/` — Microcanonical Langevin Monte Carlo (implemented from scratch;
//!              no Rust crate provides this as of Feb 2026).
//! - `nuts`   — NUTS via nuts-rs, bridged through the `LogPosterior` trait.
//! - `ensemble` — Rayon-based parallel chain runner (replaces `jax.pmap`).

pub mod ensemble;
pub mod mclmc;
pub mod nuts;

pub use ensemble::{run_chains, ChainResult};
