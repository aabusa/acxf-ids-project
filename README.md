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
- `src/model.py` — trains a Conv1D classifier with class weighting (the
  dataset is heavily imbalanced — see Results below) and saves
  `models/Model.keras` plus `Results/training_history.json`.
- `src/evaluate.py` — loads a saved model, prints a confusion matrix and
  classification report against the held-out test set, and plots training
  curves to `Results/evaluation_training_curves.png`.
- `src/predict.py` — runs the saved model + preprocessing artifacts against
  a CSV of raw, unlabeled NSL-KDD rows:

  ```bash
  python src/predict.py path/to/rows.csv --model Model
  ```

## Results

Class distribution in the training set is highly imbalanced:

| class  | train samples |
|--------|---------------|
| normal | 67,343        |
| dos    | 45,927        |
| probe  | 11,656        |
| r2l    | 995           |
| u2r    | 52            |

Current test-set performance (see `Results/` for the latest run) is strong on
`dos`/`normal`/`probe` but weak on the rare `r2l`/`u2r` classes despite class
weighting — improving recall on those two is the main open problem.

## Project layout

```
data/     raw KDDTrain+/KDDTest+ txt, preprocessed .npy tensors, fitted
          scaler/encoders (.pkl)
models/   saved .keras checkpoints
Results/  training history, evaluation plots, classification reports
src/      pipeline scripts (see above)
```
