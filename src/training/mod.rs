//! Warmstart training: deep-ensemble pretraining before MCMC sampling.
//!
//! Mirrors `src/training/trainer.py::BDETrainer::train_de_member`.

pub mod warmstart;

pub use warmstart::train_warmstart;
