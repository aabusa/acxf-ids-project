import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

feature_names = joblib.load("data/feature_names.pkl")
n_features = len(feature_names)

X_test = np.load("data/X_test.npy").reshape(-1, n_features)
X_train = np.load("data/X_train.npy").reshape(-1, n_features)
model = load_model("models/Model.keras")


def predict_fn(x):
    return model.predict(x.reshape(-1, n_features, 1), verbose=0)


background = shap.sample(X_train, 50, random_state=42)
sample = X_test[:50]

explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
shap_values = explainer(sample)

# Average |SHAP value| across samples and output classes -> one importance
# score per feature, so the plot isn't split five ways per class.
importance = np.abs(shap_values.values).mean(axis=(0, 2))
order = np.argsort(importance)[::-1][:15]

plt.figure(figsize=(8, 6))
plt.barh([feature_names[i] for i in order][::-1], importance[order][::-1])
plt.xlabel("mean(|SHAP value|)")
plt.title("Top 15 features by SHAP importance")
plt.tight_layout()
plt.savefig("Results/shap_summary.png")
print("Saved SHAP summary to Results/shap_summary.png")

for i in order:
    print(f"{feature_names[i]}: {importance[i]:.4f}")
