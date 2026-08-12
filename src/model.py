import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


X = np.load("data/X_train.npy")
y = np.load("data/y_train.npy")

# Hold out a real (non-synthetic) validation set before oversampling, so
# validation metrics aren't inflated by SMOTE-generated points.
X_train_raw, X_val, y_train_raw, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE needs 2D input; drop the size-1 channel dim used for Conv1D and
# restore it after resampling.
n_features = X_train_raw.shape[1]
X_train_res, y_train = SMOTE(random_state=42).fit_resample(
    X_train_raw.reshape(len(X_train_raw), n_features), y_train_raw
)
X_train = X_train_res.reshape(-1, n_features, 1)
print("Class counts after SMOTE:", dict(zip(*np.unique(y_train, return_counts=True))))

model = Sequential([
    Conv1D(input_shape=(41, 1), kernel_size=3, filters=64, activation='relu', padding='same'),
    Conv1D(kernel_size=3, filters=128, activation='relu', padding='same'),
    Dropout(0.3),
    Flatten(),
    Dense(units=64, activation='relu'),
    Dropout(0.3),
    Dense(units=5, activation='softmax'),
])

model.summary()

model.compile(optimizer = 'adam' ,loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

os.makedirs('models', exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('models/Model.keras', monitor='val_loss', save_best_only=True),
]

history = model.fit(X_train,y_train,epochs = 40,batch_size = 128,
                     validation_data=(X_val, y_val),callbacks=callbacks)

# Save the training history for later evaluation and plotting
os.makedirs('Results', exist_ok=True)
history_path = os.path.join('Results', 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"Saved training history to {history_path}")