//! Deep-ensemble warmstart training using Burn's optimizer.
//!
//! Trains `n_chains` independent FCN members from different random
//! initialisations.  Each trained parameter vector becomes the starting
//! position of one MCMC chain.
//!
//! Mirrors `trainer.py::train_de_member` and `single_step_regr / _class`.

use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::AutodiffBackend,
        Shape, Tensor, TensorData,
    },
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::{
    config::{MileConfig, Task, WarmstartConfig},
    models::fcn::{build_fcn, FcnModule},
    MileError,
};

// ── Progress bar styles ───────────────────────────────────────────────────────

fn member_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "warmstart [{elapsed_precise}] {bar:40.green/dim} {pos}/{len} members  {msg}",
    )
    .unwrap()
    .progress_chars("=>-")
}

fn epoch_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  member {prefix} [{bar:35.cyan/dim}] epoch {pos}/{len}  {msg}",
    )
    .unwrap()
    .progress_chars("=>-")
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Train `n_chains` independent ensemble members and return their flat
/// parameter vectors (one per chain).
///
/// Uses the same Adam optimiser and early-stopping logic as the Python
/// reference (`WarmStartConfig.patience`).
pub fn train_warmstart<B: AutodiffBackend>(
    cfg: &MileConfig,
    train_x: &Tensor<B, 2>,
    train_y: &Tensor<B, 1>,
    valid_x: &Tensor<B, 2>,
    valid_y: &Tensor<B, 1>,
    device: &B::Device,
) -> Result<Vec<Vec<f32>>, MileError> {
    let n_chains = cfg.sampler.n_chains;
    let ws_cfg = &cfg.warmstart;
    let mut init_params: Vec<Vec<f32>> = Vec::with_capacity(n_chains);

    let mp = MultiProgress::new();
    let outer = mp.add(ProgressBar::new(n_chains as u64));
    outer.set_style(member_style());

    for chain_id in 1..=n_chains {
        let epoch_pb = mp.insert_after(&outer, ProgressBar::new(ws_cfg.max_epochs as u64));
        epoch_pb.set_style(epoch_style());
        epoch_pb.set_prefix(format!("{chain_id}/{n_chains}"));

        let mut model = build_fcn::<B>(&cfg.model, device);
        let optim_cfg = AdamConfig::new().with_epsilon(1e-8);
        let mut optim = optim_cfg.init::<B, FcnModule<B>>();

        let flat = train_member(
            &mut model,
            &mut optim,
            ws_cfg,
            &cfg.task,
            train_x,
            train_y,
            valid_x,
            valid_y,
            device,
            &epoch_pb,
        )?;

        epoch_pb.finish_and_clear();
        outer.inc(1);
        outer.set_message(format!("({} params)", flat.len()));
        init_params.push(flat);
    }

    outer.finish_with_message("done");
    Ok(init_params)
}

// ── Per-member training loop ──────────────────────────────────────────────────

fn train_member<B: AutodiffBackend>(
    model: &mut FcnModule<B>,
    optim: &mut impl Optimizer<FcnModule<B>, B>,
    cfg: &WarmstartConfig,
    task: &Task,
    train_x: &Tensor<B, 2>,
    train_y: &Tensor<B, 1>,
    valid_x: &Tensor<B, 2>,
    valid_y: &Tensor<B, 1>,
    device: &B::Device,
    pb: &ProgressBar,
) -> Result<Vec<f32>, MileError> {
    let mut valid_losses: Vec<f32> = Vec::new();

    for epoch in 0..cfg.max_epochs {
        // ── Forward + backward ────────────────────────────────────────────────
        let logits = model.forward(train_x.clone());
        let loss = compute_loss::<B>(logits, train_y.clone(), task, device);

        // ── Gradient step ─────────────────────────────────────────────────────
        let grads = GradientsParams::from_grads(loss.backward(), &*model);
        *model = optim.step(cfg.learning_rate, model.clone(), grads);

        // ── Validation ────────────────────────────────────────────────────────
        // model.valid() → FcnModule<B::InnerBackend>; strip autodiff from data.
        let val_logits = model.valid().forward(valid_x.clone().inner());
        let val_loss = compute_loss_inner::<B>(val_logits, valid_y.clone().inner(), task, device);
        valid_losses.push(val_loss);

        pb.inc(1);
        if epoch % 10 == 0 {
            pb.set_message(format!("val={val_loss:.4}"));
        }

        // ── Early stopping ────────────────────────────────────────────────────
        if early_stop(&valid_losses, cfg.patience) {
            pb.set_message(format!("stopped  val={val_loss:.4}"));
            break;
        }
    }

    Ok(model.to_flat_vec())
}

// ── Loss functions ────────────────────────────────────────────────────────────

/// Compute training loss and return as an autodiff tensor (for `.backward()`).
fn compute_loss<B: AutodiffBackend>(
    logits: Tensor<B, 2>,
    y: Tensor<B, 1>,
    task: &Task,
    device: &B::Device,
) -> Tensor<B, 1> {
    match task {
        Task::Regression => gaussian_nll_loss(logits, y),
        Task::Classification { n_classes: _ } => {
            // Convert f32 labels to integer targets for cross-entropy.
            let y_int = y
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<_>>();
            let targets_data = TensorData::new(y_int, Shape::new([logits.dims()[0]]));
            let targets = Tensor::<B, 1, burn::tensor::Int>::from_data(targets_data, device);
            CrossEntropyLossConfig::new()
                .init(device)
                .forward(logits, targets)
        }
        Task::CountRegression { dispersion } => negbin_nll_loss(logits, y, *dispersion),
    }
}

/// Compute validation loss as a plain `f32` (no autodiff needed).
fn compute_loss_inner<B: AutodiffBackend>(
    logits: Tensor<B::InnerBackend, 2>,
    y: Tensor<B::InnerBackend, 1>,
    task: &Task,
    _device: &B::Device,
) -> f32 {
    match task {
        Task::Regression => {
            let mu = logits
                .clone()
                .slice([0..logits.dims()[0], 0..1])
                .squeeze::<1>();
            let diff = y - mu;
            let data = diff.powi_scalar(2).mean().into_data();
            data.to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0)
        }
        Task::Classification { .. } => {
            // Simple cross-entropy proxy: -mean(log_softmax at true class)
            // For validation we use MSE on probabilities as a proxy.
            0.0 // TODO: implement proper validation loss for classification
        }
        Task::CountRegression { dispersion } => {
            let r = *dispersion;
            let [batch, _] = logits.dims();
            let log_mu = logits.slice([0..batch, 0..1]).squeeze::<1>();
            let mu = log_mu.clone().exp();
            let log_rmu = mu.add_scalar(r).log();
            let nll = (y.clone().add_scalar(r) * log_rmu - y * log_mu).mean();
            let data = nll.into_data();
            data.to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(0.0)
        }
    }
}

/// Heteroscedastic Gaussian NLL: logits `[batch, 2]` = (mu, log_sigma).
fn gaussian_nll_loss<B: AutodiffBackend>(logits: Tensor<B, 2>, y: Tensor<B, 1>) -> Tensor<B, 1> {
    let [batch, _] = logits.dims();
    let mu = logits.clone().slice([0..batch, 0..1]).squeeze::<1>();
    let log_sigma = logits.slice([0..batch, 1..2]).squeeze::<1>();
    let sigma = log_sigma.exp().clamp(1e-6, 1e6);
    let diff = y - mu;
    let nll = (diff.powi_scalar(2) / sigma.clone().powi_scalar(2)) * 0.5
        + sigma.log()
        + (2.0 * std::f32::consts::PI).ln() * 0.5;
    nll.mean()
}

/// Negative-binomial NLL for warmstart: logits `[batch, 1]` = log_mu.
///
/// Constant terms (lgamma) are omitted here because they don't affect gradients.
/// The full constant is added inside `BnnLogPosterior::log_likelihood_count_regression`
/// for correct energy values during MCMC.
fn negbin_nll_loss<B: AutodiffBackend>(logits: Tensor<B, 2>, y: Tensor<B, 1>, r: f32) -> Tensor<B, 1> {
    let [batch, _] = logits.dims();
    let log_mu = logits.slice([0..batch, 0..1]).squeeze::<1>();
    let mu = log_mu.clone().exp();
    let log_rmu = mu.add_scalar(r).log();           // log(r + mu_i)
    (y.clone().add_scalar(r) * log_rmu - y * log_mu).mean()
}

// ── Early stopping ────────────────────────────────────────────────────────────

/// Returns `true` if none of the last `patience` validation losses improved
/// over the loss at position `-(patience+1)`.
///
/// Mirrors `trainer.py::earlystop`.
fn early_stop(losses: &[f32], patience: usize) -> bool {
    if losses.len() <= patience {
        return false;
    }
    let reference = losses[losses.len() - patience - 1];
    losses[losses.len() - patience..].iter().all(|&l| l >= reference)
}
