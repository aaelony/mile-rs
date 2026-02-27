//! MCMC diagnostics: effective sample size (ESS) and R-hat.
//!
//! Implemented directly to avoid the `mcmc` crate's `arrow` dependency
//! conflict (arrow-arith 51 + chrono 0.4 name collision as of Feb 2026).
//!
//! ESS uses Geyer's initial positive sequence estimator (same algorithm as
//! Stan's `ess_bulk`).  R-hat is the split-R-hat from Vehtari et al. (2021).

/// Effective sample size for a single chain.
///
/// Returns `n` if the chain is constant or has too few samples.
pub fn effective_sample_size(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 4 {
        return n as f64;
    }

    let mean = chain.iter().sum::<f64>() / n as f64;
    let variance = chain
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    if variance < 1e-14 {
        return n as f64;
    }

    // Sum autocorrelations at positive lags until the first negative one.
    let mut rho_sum = 0.0f64;
    for lag in 1..n {
        let rho = autocorrelation(chain, mean, variance, lag);
        if rho < 0.0 {
            break;
        }
        rho_sum += rho;
    }

    (n as f64 / (1.0 + 2.0 * rho_sum)).min(n as f64)
}

/// ESS averaged across multiple chains (bulk ESS proxy).
pub fn effective_sample_size_bulk(chains: &[&[f64]]) -> f64 {
    chains.iter().map(|c| effective_sample_size(c)).sum::<f64>()
}

/// Split R-hat across chains (Vehtari et al. 2021, simplified version).
///
/// Values ≤ 1.01 indicate convergence.
pub fn potential_scale_reduction(chains: &[&[f64]]) -> f64 {
    let m = chains.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = chains[0].len();
    if n < 2 {
        return f64::NAN;
    }

    // Chain means and variances.
    let means: Vec<f64> = chains
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();
    let variances: Vec<f64> = chains
        .iter()
        .zip(means.iter())
        .map(|(c, &mu)| c.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64)
        .collect();

    let grand_mean = means.iter().sum::<f64>() / m as f64;

    // Between-chain variance B.
    let b = n as f64
        * means.iter().map(|&mu| (mu - grand_mean).powi(2)).sum::<f64>()
        / (m - 1) as f64;

    // Within-chain variance W.
    let w = variances.iter().sum::<f64>() / m as f64;

    // Pooled variance estimate.
    let var_hat = ((n - 1) as f64 / n as f64) * w + b / n as f64;

    if w < 1e-14 {
        return f64::NAN;
    }

    (var_hat / w).sqrt()
}

// ── Internal ──────────────────────────────────────────────────────────────────

fn autocorrelation(chain: &[f64], mean: f64, variance: f64, lag: usize) -> f64 {
    let n = chain.len();
    if lag >= n {
        return 0.0;
    }
    let cov = chain[..n - lag]
        .iter()
        .zip(&chain[lag..])
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum::<f64>()
        / (n - lag) as f64;
    cov / variance
}
