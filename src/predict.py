import argparse
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description="Run IDS predictions on raw NSL-KDD formatted rows")
parser.add_argument("input_csv", help="CSV file with unlabeled rows in NSL-KDD column order (no header)")
parser.add_argument("--model", default="Model", help="Model name under models/ to load (default: Model)")
args = parser.parse_args()

scaler = joblib.load("data/scaler.pkl")
protocol_encoder = joblib.load("data/protocol_encoder.pkl")
service_encoder = joblib.load("data/service_encoder.pkl")
flag_encoder = joblib.load("data/flag_encoder.pkl")
label_encoder = joblib.load("data/label_encoder.pkl")
feature_names = joblib.load("data/feature_names.pkl")

model = load_model(f"models/{args.model}.keras")

df = pd.read_csv(args.input_csv, names=feature_names)

df["protocol_type"] = protocol_encoder.transform(df["protocol_type"])
df["service"] = service_encoder.transform(df["service"])
df["flag"] = flag_encoder.transform(df["flag"])

X_scaled = scaler.transform(df[feature_names])
X_scaled = np.expand_dims(X_scaled, axis=-1)

y_pred = model.predict(X_scaled)
predicted_classes = label_encoder.inverse_transform(y_pred.argmax(axis=1))

for i, (label, confidence) in enumerate(zip(predicted_classes, y_pred.max(axis=1))):
    print(f"row {i}: {label} (confidence {confidence:.2f})")
