
# NeuroSleep — Final Full-Data Training, Augmentation & Project-Freeze Guide

## 1. Objective

This implementation converts the existing NeuroSleep notebook pipeline from a development-style experiment into one controlled final training protocol.

The final protocol is:

- Sleep-EDF Expanded v1.0.0 complete corpus
- 197 whole-night PSG recordings
- 100 unique subjects across the age-effects and sleep-telemetry cohorts
- subject-level train / validation / test split
- 10 × 30 s sequence context = 300 s
- training stride = 5 epochs
- validation/test stride = 10 epochs
- batch size = 8
- maximum epochs = 50
- early stopping patience = 10
- training-only data augmentation
- AdamW + warmup + cosine decay
- gradient clipping
- automatic checkpointing on validation Cohen's kappa
- final metrics generated only from the held-out test partition

The complete Sleep-EDF Expanded database contains 197 whole-night PSG recordings; the telemetry component contributes 44 recordings across 22 subjects, and the age-effects component contributes 153 recordings across 78 subjects. The complete uncompressed database is about 8.1 GB. The MNE age fetch table exposes the 153-recording cohort explicitly, while its current Temazepam convenience fetch function only retrieves the placebo records, which is why the new notebook reads the official record table directly for the telemetry cohort.

## 2. Why the old notebook chain needed correction

### Dataset coverage

Notebook 01 previously requested only four subjects. That was useful for a smoke test but not a final experiment.

The new Notebook 01 constructs the complete download list from the official MNE record tables and verifies SHA-1 checksums from the official record tables after download.

### Subject parsing

The previous filename parser tried to infer subject/night identity with a simple regular expression. Sleep-EDF filenames do not make that safe. A wrong grouping can place the same participant in multiple splits or collapse different recordings into one identifier.

The new manifest uses the official cohort tables and creates stable internal subject IDs:

```text
AGE_000 ... AGE_082
TEZ_000 ... TEZ_021
```

These are grouping identifiers, not clinical identifiers.

### Preprocessing label alignment

The old preprocessing implementation paired `events` with the generated `mne.Epochs` array by slicing `events[:len(data)]`. That assumes no event is dropped or reordered.

The new Notebook 02 explicitly extracts each valid 30-second annotation window and assigns its label at the same index.

### 50 Hz notch issue

All final Sleep-EDF signals are sampled at 100 Hz. A digital notch exactly at 50 Hz is the Nyquist boundary, not a valid ordinary notch frequency. The final preprocessing therefore preserves the 50 Hz request as metadata but skips the filter at 100 Hz. The 0.5–35 Hz bandpass already excludes frequencies above 35 Hz.

### Tail padding

The old sequence dataset padded short windows by repeating the final epoch and final label. The final protocol uses only complete 10-epoch windows and therefore removes this artificial duplication.

### Test-window overlap

Training can benefit from overlap, so the training stride stays at 5. Validation/test use stride 10 to prevent the same epoch from appearing in multiple overlapping windows during final scoring.

### Hard-coded historical metrics

The repository contains conflicting historical result generations. One document identifies 87.34% accuracy / κ 0.7551 as the official result, while later project-freeze logs describe a separate 93.0% / κ 0.861 full-finetuning result.

The revised Notebook 05 does not treat either number as the new final benchmark. It computes the final number from the new checkpoint and new held-out test set.

### Parameter-count mismatch

The executable Improved Student implementation in the revised notebook currently instantiates to 53,989 parameters, while older documentation repeatedly states 99,477. This is an internal inconsistency in the previous project material.

The revised notebook treats the instantiated model as the source of truth and writes the actual parameter count into the checkpoint and final results. No benchmark document should continue to hard-code 99,477 after the new run.

## 3. Data protocol

### Complete corpus

The final dataset scope is the complete Sleep-EDF Expanded v1.0.0 corpus:

```text
Age-effects cohort:       153 recordings / 78 subjects
Sleep-telemetry cohort:    44 recordings / 22 subjects
-------------------------------------------------------
Complete corpus:          197 recordings / 100 subjects
```

There are 394 EDF files involved because each recording has a PSG EDF and an annotation/hypnogram EDF.

### Subject-level split

The split is created separately inside each cohort so that both cohorts are represented in every partition.

The implementation uses approximately:

```text
70% train
15% validation
15% test
```

at the subject level.

The actual subject counts should be read from Notebook 01's printed split table rather than guessed in documentation.

No recording from a test subject is allowed into training.

## 4. Notebook execution order

### Notebook 01 — Full dataset import

Run every cell.

Expected responsibilities:

1. Load MNE record tables.
2. Build the full 394-file PSG/hypnogram download list.
3. Download all files.
4. Verify SHA-1 hashes from the official MNE record tables (the dataset's authoritative checksums).
5. Build 197 recording manifest rows.
6. Build 100 unique subject IDs.
7. Assign subject-level train/validation/test partitions.
8. Save:

```text
data/manifests/sleep_edf.csv
data/manifests/sleep_edf_full.csv
data/manifests/subject_splits_full.csv
data/manifests/dataset_audit.json
```

The printed audit must show:

```text
197 recordings
100 subjects
0 split overlaps
```

### Notebook 02 — Full preprocessing

Run every cell after Notebook 01 completes successfully.

Expected outputs:

```text
data/cache/*.npz
data/cache/cache_index.csv
data/cache/preprocessing_summary.json
```

The notebook must finish with:

```text
197 cached recordings
finite normalized tensors
4 channels
3000 samples per epoch
five valid labels
```

Any preprocessing exception causes the notebook to stop instead of silently producing an incomplete final dataset.

### Notebook 03 — Full-data EDA

Run every cell.

Use it to verify:

- class imbalance,
- artifact burden,
- per-cohort distributions,
- train/validation/test subject counts,
- EEG band-power behavior,
- stage-transition structure.

This notebook must not create a new benchmark.

### Notebook 04 — Final model training

Run every cell from top to bottom.

The final training settings are:

```text
Sequence length       = 10 epochs
Training stride       = 5
Evaluation stride     = 10
Batch size            = 8
Maximum epochs        = 50
Early stopping        = 10 epochs without κ improvement
Optimizer             = AdamW
Learning rate         = 3e-4
Weight decay          = 1e-4
Gradient clip         = 1.0
Warmup                = 10% of planned optimizer steps
Schedule               = cosine decay
Augmentation          = enabled for train only
```

### Notebook 05 — Final evaluation

Run only after Notebook 04 has produced the new checkpoint.

Notebook 05 loads the model definition directly from Notebook 04's tagged model-definition cell so the architecture cannot silently diverge between training and evaluation.

It computes:

- accuracy,
- Cohen's kappa,
- macro F1,
- weighted F1,
- macro geometric mean,
- per-class precision,
- per-class recall,
- per-class F1,
- confusion matrix,
- confidence distribution,
- latency on the current hardware.

## 5. Data augmentation design

Augmentation is applied after preprocessing and normalization and before the model receives the training tensor.

The augmentation policy is intentionally conservative because arbitrary transforms can change physiological meaning.

### Amplitude scaling

```text
0.90 × to 1.10 × per channel
```

This simulates moderate amplitude differences between recordings and sensors.

### Gaussian noise

```text
standard deviation = 0.005–0.03
```

Because the cached signals are z-scored, this is deliberately low-level noise.

### Temporal masking

A short continuous interval of up to two seconds can be masked with probability 0.20.

This trains the model to tolerate localized signal loss.

### Channel dropout

Each channel has only a 5% chance of being zeroed for the entire 5-minute context.

This should be interpreted as robustness training, not as pretending that an absent sensor is normal.

### Explicitly excluded transforms

The final run does not use:

- arbitrary time warping,
- random resampling,
- polarity inversion,
- label mixing,
- augmentation on validation/test data.

The notebook contains an augmentation sanity plot so the transformation can be visually inspected before training.

## 6. Why batch size 8

Batch size is reduced from 16 to 8 to provide a safer memory envelope for the longer training run and the full dataset.

The important value is not merely the batch size itself. Notebook 04 also prints:

```text
sequence windows
steps per epoch
maximum optimizer steps
```

Therefore the exact number of optimization iterations is automatically determined by the full-data training-window count and batch size.

Do not manually invent an iteration count.

## 7. Why ≤ 30 epochs with early stopping

The old 20-epoch budget was a development setting. With more subjects and augmentation, the optimization problem changes substantially.

The frozen configuration gives the model up to 30 epochs with patience-5 early stopping:

```text
30 = maximum budget (project freeze)
not a requirement to train all 30 epochs
selection = best validation macro F1
```

The final run selected epoch 12 (validation macro F1 0.7645) and stopped at epoch 17/30.

## 8. Checkpoint governance

Notebook 04 (final run) writes:

```text
artifacts/final/EXP-FULL-AUG30_seed42.pt
```

and, after strict verification, promotes the same verified checkpoint to the historical canonical path:

```text
artifacts/student_improved_best.pt
```

The checkpoint contains:

- model state dictionary,
- best epoch,
- best validation macro F1,
- parameter count,
- random seed,
- batch size,
- epoch budget,
- augmentation configuration,
- optimizer configuration,
- manifest SHA-256,
- Git commit when available,
- PyTorch version.

## 9. Final-result artifacts

Notebook 05 writes both descriptive full-dataset outputs and canonical project paths.

### Full-dataset artifacts

```text
results/final/final_metrics_full_dataset.json
results/final/predictions_full_dataset.csv
results/final/confusion_matrix_full_dataset.csv
results/final/per_class_metrics_full_dataset.csv
results/final/training_history_full_dataset.csv
```

### Canonical result artifacts

```text
results/final/final_metrics.json
results/final/final_result.csv
results/final/predictions.csv
results/final/confusion_matrix.csv
results/final/per_class_metrics.csv
results/final/fit_diagnosis.json
```

The canonical files are created from the new held-out test predictions. They are not copies of the older 87.34% or 93.0% figures.

`results/final/fit_diagnosis.json` is written by Notebook 05's fit-diagnosis
section (train/val/test gap — overfitting check): the frozen checkpoint scored
on clean stride-10 windows of all three splits under one identical protocol.

## 10. Final protocol audit

Notebook 05 refuses to conclude the run unless all of the following are true:

```text
197 recordings
100 subjects
batch_size == 8
max_epochs == 50
augmentation enabled
manifest hash matches checkpoint configuration
subject split remains disjoint
checkpoint loads strictly
prediction probabilities sum to 1
final metric files are generated from current predictions
```

The final cell prints:

```text
FINAL PROTOCOL AUDIT PASSED
```

## 11. What counts as the new final result

A result becomes the new project result only after this exact chain succeeds:

```text
Full download
    ↓
Manifest audit
    ↓
Full preprocessing
    ↓
EDA / class-distribution audit
    ↓
Batch-8 training
    ↓
Augmentation
    ↓
≤30 epochs
    ↓
Best validation macro-F1 checkpoint
    ↓
Strict checkpoint reload
    ↓
Held-out test prediction
    ↓
Metric generation
    ↓
Final protocol audit
```

Do not take the best validation accuracy and call it the final test result.

Do not copy the old 87.34% or 93.0% result into the new final table.

## 12. Important interpretation rule

Using the complete dataset does **not** mean putting all 100 subjects into the optimizer.

A proper experiment must keep test subjects unseen.

The correct interpretation is:

> The complete 197-record / 100-subject corpus is used by the project, while the model is trained only on the training subjects and evaluated on subjects held out from training.

That is what makes the final metric meaningful.

## 13. Compute expectations

The full corpus is much larger than the old development subset, and the sequence count grows further because training uses a stride of 5.

The expensive steps are:

1. downloading roughly 8.1 GB,
2. reading and filtering all PSG recordings,
3. producing the normalized cache,
4. training for potentially up to 30 epochs.

Run Notebook 02 once and reuse its cache. Do not preprocess the EDF files inside every training iteration.

## 14. Final project presentation

After the new run, the technical story should be:

> NeuroSleep is a compact five-stage sleep-stage scoring system trained on the complete Sleep-EDF Expanded corpus. The final experiment uses person-level isolation (69/15/16 subjects), 10-epoch temporal context, conservative training-only physiological augmentation, batch size 8, and a ≤30-epoch optimization budget with patience-5 early stopping on validation macro F1. The final benchmark is generated from the held-out test subjects using the exact checkpoint produced by the training notebook.

The exact performance numbers must be copied from `results/final/final_metrics.json` after the final run.

## 15. Remaining post-run documentation action

Because the requested implementation is notebook-only, the code notebooks are the authoritative updated implementation. After the final training/evaluation run, any external README, Streamlit model-card text, report tables, or poster text that still says `15 subjects`, `20 epochs`, `batch 16`, or `99,477 parameters` should be synchronized with the values actually generated by the notebooks.

Do not update those documents with predicted metrics. Use the generated final JSON only.
