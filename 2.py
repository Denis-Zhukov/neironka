import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten

x_test = []
y_test = []

# 2 - [0,0,1,0,0,0,0,0,0,0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = load_model('100.h5')

pred = model.predict(x_test);
pred = np.argmax(pred, axis=1)

print(pred[:30])
print(y_test[:30])

mask = pred == y_test

print(mask[:10])

x_false = x_test[~mask]

print('Количетсво неверных определений:', x_false.shape[0])
