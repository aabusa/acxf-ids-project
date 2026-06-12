import numpy as np
from tensorflow.keras.models import load_model 
from sklearn.metrics import confusion_matrix, classification_report

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
target_names=['dos', 'normal', 'probe', 'r2l', 'u2r']

model_name = input("Enter the name of the model that you want to evaluate: ")
model = load_model(f"models/{model_name}.keras")

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
report = classification_report( y_test, y_pred.argmax(axis=1), target_names=target_names)

print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)    
