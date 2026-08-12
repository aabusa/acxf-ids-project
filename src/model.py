import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


X = np.load("data/X_train.npy")
y = np.load("data/y_train.npy")

# Both plain SMOTE and Borderline-SMOTE were tried here and dropped: r2l f1
# was 0.47 with class weighting alone vs. 0.28/0.27 oversampled, because the
# synthetic r2l points end up too close to the normal region of feature
# space. Class weighting also composes better with early stopping on a real
# (non-synthetic) validation split.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weight))

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
                     validation_data=(X_val, y_val),class_weight=class_weight_dict,callbacks=callbacks)

# Save the training history for later evaluation and plotting
os.makedirs('Results', exist_ok=True)
history_path = os.path.join('Results', 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"Saved training history to {history_path}")