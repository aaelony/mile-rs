//! Neural network models.
//!
//! Each model is provided in two forms:
//!
//! 1. **Module form** (`struct Fcn<B: AutodiffBackend>`) — used for warmstart
//!    training with Burn's optimizer.
//!
//! 2. **Pure-function form** (`fcn_forward(params, x)`) — used during MCMC
//!    sampling, where the network weights are the MCMC position and we need to
//!    differentiate through the forward pass w.r.t. those weights.
//!
//! The `params` module provides `FcnParamFlattener` which bridges between the
//! two representations.

pub mod fcn;
pub mod params;

pub use fcn::{Fcn, FcnModule, fcn_forward};
pub use params::{FcnLayerParams, FcnParamFlattener};
