import numpy as np
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

num_classes = 2
batch_size = 128
epochs = 20
(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = []
Y_train = []

X_test = []
Y_test = []

for i in range(len(y_train)):
	if y_train[i] == 0 or y_train[i] == 1:
		X_train.append(x_train[i])
		Y_train.append(y_train[i])

for i in range(len(y_test)):
	if y_test[i] == 0 or y_test[i] == 1:
		X_test.append(x_test[i])
		Y_test.append(y_test[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train.reshape(X_train.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(Y_train.shape, 'train samples')
print(Y_test.shape, 'test samples')

def neural_net(kernel_initializer):
	model = Sequential()
	model.add(Dense(1000,kernel_initializer=kernel_initializer,activation='tanh', input_shape=(784,)))
	model.add(Dense(1000,kernel_initializer=kernel_initializer, activation='tanh'))
	model.add(Dense(2,kernel_initializer=kernel_initializer, activation='sigmoid'))

	model.summary()

	model.compile(loss='MSE',
              optimizer=RMSprop(),
              metrics=['accuracy'])

	history = model.fit(X_train, Y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=(X_test, Y_test))
	
	score = model.evaluate(X_test, Y_test, verbose=0)
	
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

print("All Weights are zeroes:")
neural_net('zeros')

print("All weights are ones:")
neural_net('ones')

print("All weights are random ")
neural_net('random_uniform')

