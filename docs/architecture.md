# Liquid Neural Benchmark — Architecture Specification

## 1. Purpose

This repository provides a unified and reproducible framework for comparing:

- Closed-form Continuous-time Networks (CfC)
- Liquid Time-Constant Networks (LTC)
- GRU
- LSTM
- CNN
- TCN
- Transformer

across multiple temporal datasets and controlled robustness conditions.

The repository separates:

1. Dataset preparation
2. Model implementations
3. Training
4. Evaluation
5. Robustness perturbations
6. Statistical analysis
7. Experiment orchestration
8. Result reporting

No dataset notebook should contain duplicated training, model, metric,
evaluation, splitting, or perturbation logic.

## 2. Main Architecture

```text
Dataset Adapter
      |
      v
DataBundle and SequenceBatch
      |
      v
Experiment Runner
      |
      +-- Model Registry
      +-- Trainer
      +-- Evaluator
      +-- Checkpoint Manager
      +-- Result Writer
      +-- Environment Recorder
      |
      v
Per-run and aggregated results
```

## 3. Core Data Contracts

### SequenceBatch

Every dataset must return batches containing:

- values
- targets
- timespans when available
- observation_mask when available
- padding_mask when needed
- sequence lengths when needed
- sample identifiers

The standard input shape is:

```text
[batch_size, sequence_length, feature_count]
```

### DatasetMetadata

Every dataset adapter must provide:

- Dataset name
- Task type
- Input size
- Output size
- Class names
- Split strategy
- Sampling type
- Normalization strategy
- Train, validation, and test sizes

### DataBundle

Every dataset adapter returns:

- train_loader
- validation_loader
- test_loader
- metadata
- split manifest

The experiment runner must never load raw dataset files directly.

## 4. Dataset Adapter Responsibilities

Dataset-specific code is responsible for:

- Downloading or locating raw data
- Validating source files
- Dataset-specific preprocessing
- Creating train, validation, and test splits
- Fitting normalization on training data only
- Creating DataLoaders
- Returning a DataBundle

Shared benchmark code is responsible for:

- Model creation
- Training
- Evaluation
- Metrics
- Checkpointing
- Robustness experiments
- Statistical analysis
- Result saving

## 5. Common Model Interface

Every model receives a common set of arguments:

```python
model(
    values=values,
    timespans=timespans,
    observation_mask=observation_mask,
    padding_mask=padding_mask,
    lengths=lengths,
)
```

All models return predictions shaped as:

```text
[batch_size, output_size]
```

A wrapper may ignore arguments that are not relevant to its architecture.

## 6. Model Implementation Policy

Main liquid-model benchmark implementations:

- CfC: official ncps implementation
- LTC: official ncps implementation

Baseline implementations:

- GRU: torch.nn.GRU
- LSTM: torch.nn.LSTM
- CNN: torch.nn.Conv1d
- Transformer: torch.nn.TransformerEncoder

The repository adds thin input-output wrappers only.
It must not silently rewrite the CfC or LTC equations.

Every result records:

- Implementation source
- Dependency version
- Git commit
- Model configuration
- Trainable parameter count

## 7. Model Registry

Models are created centrally through a registry.

```text
gru
lstm
cfc
ltc
cnn
tcn
transformer
```

Notebooks must not instantiate architecture classes directly.

## 8. Configuration System

Every experiment is configuration-driven.

The configuration hierarchy contains:

- Experiment configuration
- Dataset configuration
- Model configuration
- Training configuration
- Evaluation configuration
- Robustness configuration
- Reproducibility configuration

No final experiment may depend on hidden notebook variables.

## 9. Training Responsibilities

The trainer is responsible for:

- Moving batches to the selected device
- Validating batches
- Forward propagation
- Loss calculation
- Backpropagation
- Gradient clipping
- Optimizer and scheduler steps
- Validation
- Early stopping
- Checkpoint saving
- Runtime measurement
- Training-history recording

The trainer must not:

- Create dataset splits
- Tune using test data
- Select thresholds using test data
- Change experiment hyperparameters silently

## 10. Evaluation Responsibilities

The evaluator is responsible for:

- Generating predictions
- Computing task-specific metrics
- Selecting binary thresholds using validation data only
- Applying the frozen threshold to test data
- Producing confusion matrices
- Measuring inference time
- Returning structured results

Binary classification metrics include ROC-AUC, PR-AUC, precision, recall,
F1, specificity, balanced accuracy, MCC, and calibration metrics.

Multiclass metrics include accuracy, balanced accuracy, macro precision,
macro recall, macro-F1, weighted F1, MCC, and per-class F1.

Regression metrics include MAE, RMSE, R-squared, and MAPE when valid.

## 11. Experiment Runner

The ExperimentRunner coordinates a complete experiment:

1. Validate configuration
2. Record environment information
3. Set the random seed
4. Validate dataset splits
5. Build the requested model
6. Count model parameters
7. Build loss, optimizer, and scheduler
8. Train the model
9. Restore the best validation checkpoint
10. Select thresholds using validation data
11. Evaluate the test set
12. Save metrics and predictions
13. Save configuration and environment provenance

Failed runs must be recorded explicitly and never silently ignored.

## 12. Multi-Seed Experiments

Final experiments use matched random seeds for every model.

Required summary statistics include:

- Mean
- Standard deviation
- Median
- 95 percent confidence interval
- Successful-run count
- Failed-run count

## 13. Robustness Framework

Supported perturbations include:

- Gaussian noise
- Observation missingness
- Timestep removal
- Sampling-rate reduction
- Timestamp jitter
- Channel dropout
- Distribution shift

Every perturbation must:

- Preserve labels
- Be deterministic for a fixed seed
- Record exact parameters
- Support a zero-perturbation identity condition
- Avoid modifying the original batch in place

## 14. Leakage Prevention

Automated validation must detect:

- Subject overlap
- Sample-ID overlap
- Window overlap across splits
- Normalization fitted outside training data
- Test-set threshold selection
- Test-set hyperparameter tuning
- Duplicate samples
- Inconsistent class mappings

Final experiments must stop when leakage checks fail.

## 15. Result Structure

Each run stores:

```text
results/runs/<experiment_id>/<model>/seed_<seed>/
    config.yaml
    environment.json
    metrics.json
    predictions.csv
    history.csv
    split_manifest.json
    checkpoint.pt
```

Aggregated results are stored under:

```text
results/summaries/
results/statistical_tests/
results/figures/
```

## 16. Notebook Policy

Notebooks are execution and visualization interfaces only.

They may contain:

- Repository installation
- Configuration selection
- Dataset download authorization
- Experiment execution
- Result display
- Final visualizations

They must not contain:

- Model definitions
- Training loops
- Metric implementations
- Dataset splitting logic
- Repeated perturbation code

## 17. Testing Policy

The test suite must cover:

- Reproducibility
- Batch validation
- Dataset metadata
- Split overlap
- Train-only normalization
- Model output shapes
- Metric correctness
- Threshold selection
- Perturbation determinism
- Zero-perturbation identity
- Result writing
- Configuration validation

## 18. Frozen Decisions

1. Keep training as the training package name.
2. Keep perturbations as the perturbation package name.
3. All datasets produce SequenceBatch objects.
4. Dataset-specific logic remains outside the benchmark engine.
5. All models use one common forward interface.
6. Models are created through a registry.
7. Thresholds are selected using validation data only.
8. Test data is never used for tuning.
9. Official implementations are wrapped, not rewritten.
10. Every final experiment is configuration-driven.
11. Final comparisons use matched seeds.
12. Every run records code and environment provenance.
13. Leakage checks are mandatory.
14. Perturbations are deterministic and reusable.
15. Notebooks do not contain benchmark implementation code.

## 19. Development Sequence

### Milestone 1 — Data contracts

1. SequenceBatch
2. DatasetMetadata
3. DataBundle
4. DatasetAdapter
5. Split validation
6. Normalization validation

### Milestone 2 — Baseline pipeline

1. GRU wrapper
2. LSTM wrapper
3. Model registry
4. Loss factory
5. Metrics
6. Trainer
7. Evaluator
8. Single-run experiment

### Milestone 3 — Liquid models

1. Pin the official ncps dependency
2. Verify the CfC API
3. Verify the LTC API
4. Add thin wrappers
5. Add shape and timestamp tests

### Milestone 4 — First complete dataset

1. PAMAP2 adapter
2. One-seed smoke experiment
3. Leakage validation
4. Multi-seed clean benchmark
5. Result aggregation

### Milestone 5 — Additional datasets

1. PhysioNet 2012
2. Event MNIST
3. UCI HAR
4. Solar forecasting

### Milestone 6 — Robustness and analysis

1. Noise
2. Missingness
3. Downsampling
4. Timestamp jitter
5. Channel dropout
6. Distribution shifts
7. Statistical comparisons
8. Capacity-matched comparisons
9. Efficiency analysis
10. Liquid-dynamics analysis
