import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D , Flatten , Dense , Input

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")


model = Sequential([
    Input(input_shape=(41, 1)),
    Conv1D ( kernel_size = 3 ,filters = 64, activation='relu'),
    Flatten(),
    Dense(units=5, activation= 'softmax'),
    
])

model.summary()