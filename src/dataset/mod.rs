//! Data loading utilities.
//!
//! Currently supports tabular data from CSV, matching the Python `TabularLoader`.
//! Image and text loaders are stubs for future implementation.

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use std::path::Path;


// ── Tabular dataset ───────────────────────────────────────────────────────────

/// An in-memory tabular dataset (features + labels).
pub struct TabularDataset {
    /// Feature matrix, shape `[n_samples, n_features]`.
    pub features: Vec<Vec<f32>>,
    /// Labels:
    ///   - Regression: `f32` target values.
    ///   - Classification: `f32`-cast integer class indices.
    pub labels: Vec<f32>,
}

impl TabularDataset {
    /// Load a tabular dataset where the last column is the label.
    ///
    /// The delimiter is auto-detected from the first data line:
    /// comma-separated (`.csv`) or whitespace-separated (`.data` / `.txt`).
    /// An optional header row is skipped when the first line cannot be parsed
    /// as numbers.
    pub fn from_csv(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.lines().peekable();

        // ── Detect delimiter from the first non-empty line ────────────────────
        let first = loop {
            match lines.peek() {
                Some(l) if !l.trim().is_empty() => break *l,
                Some(_) => { lines.next(); }
                None => {
                    return Ok(Self { features: vec![], labels: vec![] });
                }
            }
        };
        let use_comma = first.contains(',');

        let split_line = |line: &str| -> Vec<f32> {
            if use_comma {
                line.split(',')
                    .map(|s| s.trim().parse::<f32>().unwrap_or(f32::NAN))
                    .collect()
            } else {
                line.split_ascii_whitespace()
                    .map(|s| s.parse::<f32>().unwrap_or(f32::NAN))
                    .collect()
            }
        };

        let mut features = Vec::new();
        let mut labels = Vec::new();
        let mut header_skipped = false;

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let cols = split_line(line);
            // Skip a header row: any row where the first token is non-numeric.
            if !header_skipped {
                header_skipped = true;
                let first_token = if use_comma {
                    line.split(',').next().unwrap_or("").trim().to_string()
                } else {
                    line.split_ascii_whitespace().next().unwrap_or("").to_string()
                };
                if first_token.parse::<f32>().is_err() {
                    continue; // skip header
                }
            }
            if cols.len() < 2 {
                continue;
            }
            labels.push(*cols.last().unwrap());
            features.push(cols[..cols.len() - 1].to_vec());
        }

        Ok(Self { features, labels })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Feature dimensionality.
    pub fn n_features(&self) -> usize {
        self.features.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Convert features to a Burn tensor of shape `[n_samples, n_features]`.
    pub fn features_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        let flat: Vec<f32> = self.features.iter().flatten().copied().collect();
        let [n, d] = [self.len(), self.n_features()];
        let data = TensorData::new(flat, Shape::new([n, d]));
        Tensor::from_data(data, device)
    }

    /// Convert labels to a Burn tensor of shape `[n_samples]`.
    pub fn labels_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        let data = TensorData::new(self.labels.clone(), Shape::new([self.len()]));
        Tensor::from_data(data, device)
    }

    /// Persist to CSV (features first, label last column, header included).
    ///
    /// The resulting file is readable by `from_csv`.
    pub fn to_csv(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(path)?;

        // Header
        let d = self.n_features();
        let header: Vec<String> = (0..d)
            .map(|i| format!("feature_{i}"))
            .chain(std::iter::once("label".to_string()))
            .collect();
        writeln!(file, "{}", header.join(","))?;

        // Data rows
        for (row, &label) in self.features.iter().zip(self.labels.iter()) {
            let vals: Vec<String> = row
                .iter()
                .map(|v| v.to_string())
                .chain(std::iter::once(label.to_string()))
                .collect();
            writeln!(file, "{}", vals.join(","))?;
        }

        Ok(())
    }

    /// Random 80/10/10 train/valid/test split.
    pub fn split_train_valid_test(
        self,
        seed: u64,
    ) -> (TabularDataset, TabularDataset, TabularDataset) {
        use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let n = self.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let n_train = (n as f32 * 0.8) as usize;
        let n_valid = (n as f32 * 0.1) as usize;

        let split = |idx: &[usize]| TabularDataset {
            features: idx.iter().map(|&i| self.features[i].clone()).collect(),
            labels: idx.iter().map(|&i| self.labels[i]).collect(),
        };

        let train = split(&indices[..n_train]);
        let valid = split(&indices[n_train..n_train + n_valid]);
        let test = split(&indices[n_train + n_valid..]);

        (train, valid, test)
    }
}
