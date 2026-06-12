import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D , Flatten , Dense 
from sklearn.utils.class_weight import compute_class_weight


X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")
unique_classes = np.unique(y_train)

class_weight = compute_class_weight(class_weight='balanced',
    classes= unique_classes,
    y=y_train )

class_weight_dict = dict(zip(unique_classes, class_weight))

model = Sequential([
    
    Conv1D ( input_shape=(41, 1),kernel_size = 3 ,filters = 64, activation='relu'),
    Flatten(),
    Dense(units=5, activation= 'softmax'),
    
])

model.summary()

model.compile(optimizer = 'adam' ,loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

history = model.fit(X_train,y_train,epochs = 10,batch_size = 128,validation_split=0.2,class_weight=class_weight_dict)

# Save the training history for later evaluation and plotting
os.makedirs('Results', exist_ok=True)
history_path = os.path.join('Results', 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"Saved training history to {history_path}")


model.save("models/Model.keras")