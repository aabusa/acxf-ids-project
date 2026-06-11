import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D , Flatten , Dense 

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")


model = Sequential([
    
    Conv1D ( input_shape=(41, 1),kernel_size = 3 ,filters = 64, activation='relu'),
    Flatten(),
    Dense(units=5, activation= 'softmax'),
    
])

model.summary()

model.compile(optimizer = 'adam' ,loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

model_v1 = model.fit(X_train,y_train,epochs = 10,batch_size = 128,validation_split=0.2)

model.save("models/Model_v1.keras")