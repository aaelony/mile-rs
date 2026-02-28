//! Configuration structs mirroring the Python YAML config layer.
//!
//! Serialisation order: TOML is preferred; YAML is accepted for compatibility
//! with the Python experiment files.

use serde::{Deserialize, Serialize};

// ── Data ──────────────────────────────────────────────────────────────────────

/// Data-source configuration (mirrors Python `data:` section).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Path to the CSV data file.  Relative paths are resolved from the
    /// experiment directory.  Required unless `use_synthetic = true`.
    pub path: Option<String>,
    /// When `true`, ignore `path` and generate a synthetic Gaussian dataset
    /// (`y = x[0] + noise`) with `input_dim` features.  Set explicitly in
    /// the TOML; the program will not fall back to synthetic data silently.
    #[serde(default)]
    pub use_synthetic: bool,
    /// Z-score normalise features before training/sampling.
    pub normalize: bool,
    /// Fraction of data for training (default 0.8).
    pub train_split: f32,
    /// Fraction for validation (default 0.1).
    pub valid_split: f32,
    /// Fraction for test (default 0.1).
    pub test_split: f32,
    /// Cap the dataset at this many rows.  `None` → use all.
    pub datapoint_limit: Option<usize>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            path: None,
            use_synthetic: false,
            normalize: true,
            train_split: 0.8,
            valid_split: 0.1,
            test_split: 0.1,
            datapoint_limit: None,
        }
    }
}

// ── Backend ───────────────────────────────────────────────────────────────────

/// Which Burn compute backend to use.
///
/// - `"cpu"` → `Autodiff<NdArray<f32>>`: pure CPU, chains run in parallel via Rayon.
/// - `"gpu"` → `Autodiff<Wgpu>`: GPU-accelerated forward/backward passes, chains
///   run sequentially (Rayon would contend on the single GPU command queue).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Gpu,
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::Cpu
    }
}

// ── Sampler ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SamplerKind {
    Mclmc,
    Nuts,
}

impl Default for SamplerKind {
    fn default() -> Self {
        Self::Mclmc
    }
}

/// Full sampler configuration (mirrors Python `training.sampler` section).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub sampler: SamplerKind,
    /// Number of independent chains.
    pub n_chains: usize,
    /// Samples to draw per chain after warmup.
    pub n_samples: usize,
    /// Only save every nth sample (1 = no thinning).
    pub n_thinning: usize,
    /// Warmup (adaptation) steps.
    pub warmup_steps: usize,

    // ── MCLMC-specific ────────────────────────────────────────────────────────
    pub step_size_init: f32,
    pub desired_energy_var_start: f32,
    pub desired_energy_var_end: f32,
    pub trust_in_estimate: f32,
    pub num_effective_samples: usize,
    pub diagonal_preconditioning: bool,

    // ── NUTS-specific (nuts-rs) ───────────────────────────────────────────────
    pub nuts_max_depth: u64,
    pub nuts_target_accept: f64,

    // ── Prior ─────────────────────────────────────────────────────────────────
    pub prior: PriorConfig,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerKind::Mclmc,
            n_chains: 4,
            n_samples: 1000,
            n_thinning: 1,
            warmup_steps: 1000,
            step_size_init: 0.005,
            desired_energy_var_start: 5e-4,
            desired_energy_var_end: 1e-4,
            trust_in_estimate: 1.5,
            num_effective_samples: 100,
            diagonal_preconditioning: false,
            nuts_max_depth: 10,
            nuts_target_accept: 0.80,
            prior: PriorConfig::default(),
        }
    }
}

// ── Prior ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "name", rename_all = "PascalCase")]
pub enum PriorConfig {
    StandardNormal,
    Normal { loc: f32, scale: f32 },
    Laplace { loc: f32, scale: f32 },
}

impl Default for PriorConfig {
    fn default() -> Self {
        Self::StandardNormal
    }
}

// ── Task ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Regression,
    Classification { n_classes: usize },
    /// Negative-binomial count regression with fixed dispersion `r`.
    ///
    /// The network outputs a single logit interpreted as `log(μ)` (log expected count).
    /// The dispersion parameter `r` is a fixed hyperparameter set in the config.
    ///
    /// TOML: `task = { count_regression = { dispersion = 10.0 } }`
    CountRegression { dispersion: f32 },
}

// ── Warmstart ─────────────────────────────────────────────────────────────────

/// Deep-ensemble pretraining configuration (mirrors Python `training.warmstart` section).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartConfig {
    /// Whether to perform warmstart training at all.
    pub enabled: bool,
    pub max_epochs: usize,
    /// None → full-batch gradient descent.
    pub batch_size: Option<usize>,
    pub learning_rate: f64,
    /// Early-stopping patience (epochs).
    pub patience: usize,
}

impl Default for WarmstartConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_epochs: 500,
            batch_size: None,
            learning_rate: 1e-3,
            patience: 20,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// Fully-connected network configuration (mirrors Python `model:` section).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FcnConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden layer sizes (including the final output layer).
    /// e.g. `[64, 64, 1]` → two hidden layers of 64 then output dim 1.
    pub hidden_structure: Vec<usize>,
    pub use_bias: bool,
}

impl Default for FcnConfig {
    fn default() -> Self {
        Self {
            input_dim: 10,
            hidden_structure: vec![64, 64, 1],
            use_bias: true,
        }
    }
}

impl FcnConfig {
    /// The output dimension of the final layer.
    pub fn output_dim(&self) -> usize {
        *self.hidden_structure.last().expect("hidden_structure must be non-empty")
    }

    /// Layer-by-layer (in_dim, out_dim) pairs.
    pub fn layer_shapes(&self) -> Vec<(usize, usize)> {
        let mut shapes = Vec::new();
        let mut prev = self.input_dim;
        for &h in &self.hidden_structure {
            shapes.push((prev, h));
            prev = h;
        }
        shapes
    }
}

// ── Top-level ─────────────────────────────────────────────────────────────────

/// Root configuration for a full MILE experiment.
///
/// TOML example:
/// ```toml
/// experiment_name = "diagnostics_mclmc"
/// seed = 42
/// output_dir = "results/diagnostics"
///
/// [data]
/// path = "data/airfoil.csv"
/// normalize = true
/// train_split = 0.7
/// valid_split = 0.1
/// test_split = 0.2
///
/// [model]
/// input_dim = 5
/// hidden_structure = [64, 64, 1]
/// use_bias = true
///
/// [warmstart]
/// enabled = true
/// max_epochs = 500
/// learning_rate = 1e-3
/// patience = 20
///
/// [sampler]
/// sampler = "mclmc"
/// n_chains = 4
/// n_samples = 1000
/// warmup_steps = 1000
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MileConfig {
    pub experiment_name: String,
    pub task: Task,
    /// Global RNG seed (top-level, like Python's `rng` field).
    /// Chain `i` uses `seed ^ i` for independent randomness.
    pub seed: u64,
    /// Compute backend.  `"cpu"` runs chains in parallel via Rayon;
    /// `"gpu"` runs chains sequentially on the Wgpu device.
    #[serde(default)]
    pub backend: BackendKind,
    pub data: DataConfig,
    pub model: FcnConfig,
    pub warmstart: WarmstartConfig,
    pub sampler: SamplerConfig,
    /// Directory to write samples and checkpoints.
    pub output_dir: String,
}

impl Default for MileConfig {
    fn default() -> Self {
        Self {
            experiment_name: "experiment".into(),
            task: Task::Regression,
            seed: 42,
            backend: BackendKind::default(),
            data: DataConfig::default(),
            model: FcnConfig::default(),
            warmstart: WarmstartConfig::default(),
            sampler: SamplerConfig::default(),
            output_dir: "results".into(),
        }
    }

}





