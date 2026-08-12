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
- `src/model.py` — holds out a stratified validation split, trains a Conv1D
  classifier with class weighting (the dataset is heavily imbalanced — see
  Results below, and why SMOTE oversampling was tried and dropped) and
  early stopping, then adversarially fine-tunes on FGSM examples mixed
  with clean data (see Results — this has a real caveat), and saves
  `models/Model.keras` plus `Results/training_history.json`.
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
- `src/adversarial.py` — FGSM and PGD (10-step) L-inf attacks against the
  model over a range of epsilons, saved to
  `Results/adversarial_robustness.png`.

## Results

Class distribution in the training set is highly imbalanced:

| class  | train samples |
|--------|---------------|
| normal | 67,343        |
| dos    | 45,927        |
| probe  | 11,656        |
| r2l    | 995           |
| u2r    | 52            |

Current test-set performance (2-conv-layer model with dropout, class
weighting, and early stopping on val_loss against a real stratified
validation split):

| class  | precision | recall | f1-score |
|--------|-----------|--------|----------|
| dos    | 0.97      | 0.85   | 0.90     |
| normal | 0.71      | 0.96   | 0.82     |
| probe  | 0.78      | 0.63   | 0.70     |
| r2l    | 0.93      | 0.23   | 0.36     |
| u2r    | 0.21      | 0.33   | 0.26     |

**Three imbalance strategies were tried; class weighting won on `r2l` in
every run:**

| approach                    | r2l f1 | u2r f1 | macro F1 |
|------------------------------|--------|--------|----------|
| class weighting (this model) | 0.36   | 0.26   | 0.61     |
| class weighting (earlier run)| 0.47   | 0.44   | 0.64     |
| plain SMOTE                  | 0.28   | 0.49   | 0.64     |
| Borderline-SMOTE              | 0.27   | 0.34   | 0.59     |

Both SMOTE variants left `r2l` worse than either class-weighting run —
their synthetic `r2l` points sit too close to the `normal` region of
feature space to help separate the two, and Borderline-SMOTE's boundary-
focused sampling didn't fix that. Class weighting was kept as the final
approach. Note the two class-weighting rows are the *same* code, different
runs — `u2r` has only 67 test samples, so its score swings a lot between
runs; treat any single point estimate here cautiously rather than reading
too much into small differences. SMOTE combined with class weighting isn't
a useful next step: once SMOTE balances the classes, `class_weight='balanced'`
computes to ~1.0 for every class, i.e. a no-op.

SHAP (`Results/shap_summary.png`) shows the model relies most heavily on
the `dst_host_*`/`*serror_rate`/`*rerror_rate` connection-error and
same-service-rate features — consistent with how these attack categories
actually manifest (e.g. scan/DoS traffic driving up error and repeat-service
rates), which is a reasonable sanity check that the model learned real
signal rather than spurious correlations.

**Adversarial robustness** (`Results/adversarial_robustness.png`, 2000 test
samples). `model.py` now does adversarial fine-tuning by default: after
normal training it generates FGSM examples at epsilon=0.05 against the
just-trained model and continues training on clean+adversarial data mixed
together for 10 more epochs (fixed epochs, not early-stopped on clean
val_loss, since the goal is robustness, not clean loss). Clean accuracy
cost: macro F1 0.61 -> 0.58, a modest trade.

| epsilon | undefended FGSM | undefended PGD | defended FGSM | defended PGD |
|---------|-----------------|-----------------|----------------|--------------|
| 0.01    | 0.54            | 0.39            | 0.78           | 0.78         |
| 0.02    | 0.48            | 0.30            | 0.77           | 0.76         |
| 0.05    | 0.43            | 0.26            | 0.68           | 0.37         |
| 0.10    | 0.40            | 0.23            | 0.49           | **0.09**     |
| 0.20    | 0.38            | 0.16            | 0.28           | **0.04**     |

Near the training epsilon (<=0.05) this is a large, real improvement —
PGD accuracy at epsilon=0.01 goes from 0.39 to 0.78. But **beyond the
trained epsilon, the defended model is worse against PGD than the
undefended one** (0.09 vs 0.23 at epsilon=0.1, 0.04 vs 0.16 at
epsilon=0.2). This is a known failure mode of single-step FGSM
adversarial training (Tramèr et al., "Ensemble Adversarial Training"):
the model overfits to the specific FGSM perturbation direction it was
fine-tuned on rather than learning a genuinely more robust decision
boundary, so a stronger multi-step attack (PGD) at a larger epsilon can
exploit that overfitting and do *more* damage than on the undefended
model. Reporting only the epsilon=0.01-0.05 numbers would be misleading —
**this defense should not be trusted against an adversary who isn't
epsilon-constrained to the training value.** Training directly on PGD
examples (not just FGSM) is the standard fix and would be the next thing
to try.

## Project layout

```
data/     raw KDDTrain+/KDDTest+ txt, preprocessed .npy tensors, fitted
          scaler/encoders (.pkl)
models/   saved .keras checkpoints
Results/  training history, evaluation plots, classification reports
src/      pipeline scripts (see above)
```
