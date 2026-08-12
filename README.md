# acxf-ids-project

A network intrusion detection classifier trained on the NSL-KDD dataset. A 1D
convolutional network takes the 41 KDD connection features and classifies
each connection as `normal` or one of four attack categories: `dos`, `probe`,
`r2l`, `u2r`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `KDDTrain+.txt` and `KDDTest+.txt` (NSL-KDD, no header row) in `data/`.

## Pipeline

Run in order from the project root:

```bash
python src/preprocess.py   # build train/test tensors + fit encoders/scaler
python src/model.py        # train the Conv1D model
python src/evaluate.py     # confusion matrix, per-class report, training curves
```

- `src/explore_data.py` — quick look at the raw dataset (shape, label counts).
- `src/preprocess.py` — label-encodes `protocol_type`/`service`/`flag`, maps
  the ~40 raw attack labels to the 5 classes above, Min-Max scales the
  features, and writes `data/X_train.npy`, `data/y_train.npy`,
  `data/X_test.npy`, `data/y_test.npy`. Also persists the fitted
  `scaler.pkl`, the three categorical encoders, `label_encoder.pkl`, and
  `feature_names.pkl` to `data/` so inference can reuse the exact same
  preprocessing.
- `src/model.py` — holds out a validation split, SMOTE-oversamples the
  minority classes in the training portion (the dataset is heavily
  imbalanced — see Results below), trains a Conv1D classifier with early
  stopping, and saves `models/Model.keras` plus
  `Results/training_history.json`.
- `src/evaluate.py` — loads a saved model, prints a confusion matrix and
  classification report against the held-out test set, and plots training
  curves to `Results/evaluation_training_curves.png`.
- `src/predict.py` — runs the saved model + preprocessing artifacts against
  a CSV of raw, unlabeled NSL-KDD rows:

  ```bash
  python src/predict.py path/to/rows.csv --model Model
  ```

- `src/explain.py` — SHAP (model-agnostic `PermutationExplainer`) feature
  importance on a sample of the test set, saved to
  `Results/shap_summary.png`.

## Results

Class distribution in the training set is highly imbalanced:

| class  | train samples |
|--------|---------------|
| normal | 67,343        |
| dos    | 45,927        |
| probe  | 11,656        |
| r2l    | 995           |
| u2r    | 52            |

Current test-set performance (2-conv-layer model with dropout, SMOTE
oversampling of the minority classes in the training split, and early
stopping on val_loss):

| class  | precision | recall | f1-score |
|--------|-----------|--------|----------|
| dos    | 0.96      | 0.82   | 0.88     |
| normal | 0.68      | 0.97   | 0.80     |
| probe  | 0.85      | 0.65   | 0.73     |
| r2l    | 0.96      | 0.17   | 0.28     |
| u2r    | 0.56      | 0.43   | 0.49     |

Macro-avg F1 is flat at 0.64 versus the earlier class-weighted-only model,
but the gains moved around: `dos`/`probe`/`u2r` improved, while `r2l`
recall actually dropped (0.32 -> 0.17) — 2362 of 2885 `r2l` test samples
are now misclassified as `normal`. SMOTE's synthetic `r2l` points appear to
sit too close to the `normal` region of feature space to help separate
them, which class weighting (a loss-level reweighting, not a feature-space
fix) didn't run into in the same way. Combining SMOTE with class weighting,
or using a variant like Borderline-SMOTE/SMOTE-Tomek that's more careful
near class boundaries, would be the next thing to try for `r2l`
specifically.

SHAP (`Results/shap_summary.png`) shows the model relies most heavily on
the `dst_host_*`/`*serror_rate`/`*rerror_rate` connection-error and
same-service-rate features — consistent with how these attack categories
actually manifest (e.g. scan/DoS traffic driving up error and repeat-service
rates), which is a reasonable sanity check that the model learned real
signal rather than spurious correlations.

## Project layout

```
data/     raw KDDTrain+/KDDTest+ txt, preprocessed .npy tensors, fitted
          scaler/encoders (.pkl)
models/   saved .keras checkpoints
Results/  training history, evaluation plots, classification reports
src/      pipeline scripts (see above)
```
