import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ADV_EPSILON = 0.05  # matches one of the epsilons probed in src/adversarial.py
ADV_FINE_TUNE_EPOCHS = 10

# Categorical/binary feature indices (0-indexed, matching preprocess.py's
# col_names order with "label"/"difficulty" dropped) that FGSM must never
# perturb: protocol_type, service, flag, land, logged_in, root_shell,
# su_attempted, is_host_login, is_guest_login. Perturbing these to fractional
# values doesn't correspond to any real network state.
FROZEN_FEATURE_INDICES = [1, 2, 3, 6, 11, 13, 14, 20, 21]
PERTURB_MASK = np.ones((41, 1), dtype=np.float32)
PERTURB_MASK[FROZEN_FEATURE_INDICES, 0] = 0.0


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


loss_fn = SparseCategoricalCrossentropy()


def fgsm_batch(x, y, epsilon, batch_size=512):
    adv_batches = []
    for i in range(0, len(x), batch_size):
        xb = tf.convert_to_tensor(x[i:i + batch_size])
        yb = tf.convert_to_tensor(y[i:i + batch_size])
        with tf.GradientTape() as tape:
            tape.watch(xb)
            loss = loss_fn(yb, model(xb, training=False))
        grad = tape.gradient(loss, xb)
        perturbation = epsilon * tf.sign(grad) * PERTURB_MASK
        adv_batches.append(tf.clip_by_value(xb + perturbation, 0.0, 1.0).numpy())
    return np.concatenate(adv_batches)


X_train_adv = fgsm_batch(X_train, y_train, ADV_EPSILON)
X_train_combined = np.concatenate([X_train, X_train_adv])
y_train_combined = np.concatenate([y_train, y_train])

adv_history = model.fit(X_train_combined, y_train_combined, epochs=ADV_FINE_TUNE_EPOCHS,
                         batch_size=128, validation_data=(X_val, y_val),
                         class_weight=class_weight_dict)

model.save('models/Model.keras')

for key in history.history:
    history.history[key] += adv_history.history[key]

# Save the training history for later evaluation and plotting
os.makedirs('Results', exist_ok=True)
history_path = os.path.join('Results', 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"Saved training history to {history_path}")