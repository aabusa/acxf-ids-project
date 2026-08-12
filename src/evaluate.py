import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser(description="Evaluate a saved IDS model on the test set")
parser.add_argument("--model", default="Model", help="Model name under models/ to evaluate (default: Model)")
args = parser.parse_args()

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
target_names=['dos', 'normal', 'probe', 'r2l', 'u2r']

model = load_model(f"models/{args.model}.keras")

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
report = classification_report( y_test, y_pred.argmax(axis=1), target_names=target_names)

print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

# Load and plot training history if available
history_path = os.path.join('Results', 'training_history.json')
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'o-', label='Train loss')
    plt.plot(epochs, history['val_loss'], 'o-', label='Validation loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'o-', label='Train accuracy')
    plt.plot(epochs, history['val_accuracy'], 'o-', label='Validation accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join('Results', 'evaluation_training_curves.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved evaluation training curves to {plot_path}")
else:
    print(f"No training history found at {history_path}. Train with `src/model.py` first.")

