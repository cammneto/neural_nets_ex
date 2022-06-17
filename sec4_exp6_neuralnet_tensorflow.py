import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)= mnist.load_data()

#plt.imshow(x_train[0], cmap='gray')
#plt.title('Class: ' + str(y_train[0]))
#plt.show()

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

#print(x_train[0])

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

network = Sequential()
network.add(Dense(input_shape=(784,), units=397, activation='relu'))
network.add(Dense(units=397, activation='relu'))
network.add(Dense(units=10, activation='softmax'))
network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = network.fit(x_train, y_train, batch_size=128, epochs=10)

plt.plot(history.history['loss'])
#plt.show()
plt.plot(history.history['accuracy'])
#plt.show()

accuracy_test = network.evaluate(x_test, y_test)
predictions = network.predict(x_test)

plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.title('Class: ' + str(y_test[0]))
plt.show()
