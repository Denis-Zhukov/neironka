import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

# 2 - [0,0,1,0,0,0,0,0,0,0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


def trainWithHiddenLayer(count):
    start_time = time.time()
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(count, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # SGD - Стохратический Градиентный Спуск
    # Backpropagation auto
    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

    model.evaluate(x_test, y_test_cat)

    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)

    print(pred[:30])
    print(y_test[:30])

    mask = pred == y_test
    print(mask[:10])

    x_false = x_test[~mask]

    print('Количетсво неверных определений:', x_false.shape[0])
    model.save("%s.h5" % (count));
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.show()
    return [x_false.shape[0], time.time() - start_time];


res = {25: trainWithHiddenLayer(25),
       50: trainWithHiddenLayer(50),
       75: trainWithHiddenLayer(75),
       100: trainWithHiddenLayer(100),
       125: trainWithHiddenLayer(125)}

print("%25s%40s%30s" % ("Слоёв на скрытом слоё", "Неверно определенных значений", "Время выполнения(сек.)"))
for key in res.keys():
    print("%14s%40s%39s" % (key, res.get(key)[0], res.get(key)[1]))

plt.plot(res.keys(), [(10000 - err[0]) for err in res.values()]);
plt.show();
