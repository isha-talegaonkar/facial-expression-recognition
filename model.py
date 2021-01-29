from __future__ import print_function
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 8

with open("/home/ubantu/Desktop/FER/fer2013.csv") as f:
	content = f.readlines()
 
lines = np.array(content)
 
num_of_instances = lines.size
print("number of instances: ",num_of_instances)

x_train, y_train, x_test, y_test = [], [], [], []
 
for i in range(1,num_of_instances):
	try:
  		emotion, img, usage = lines[i].split(",")
 
		val = img.split(" ")
		pixels = np.array(val, 'float32')
		 
		emotion = keras.utils.to_categorical(emotion, num_classes)
 
  		if 'Training' in usage:
   			y_train.append(emotion)
  			x_train.append(pixels)
  		elif 'PublicTest' in usage:
   			y_test.append(emotion)
   			x_test.append(pixels)
 	except:
  		print("", end="")

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
 
#1st convolution layer
model.add(Conv2D(64, (4, 4), activation='relu', padding="same", input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(4,4), strides=(2, 2)))
 
#2nd convolution layer
model.add(Conv2D(filters=64, kernel_size=(4, 4),strides=(1,1), padding="same", activation='relu'))
BatchNormalization()
model.add(Conv2D(64, kernel_size=(4, 4),strides=(1,1), padding="same", activation='relu'))
BatchNormalization()
model.add(MaxPooling2D(pool_size=(4,4), strides=(2, 2)))
model.add(Dropout(0.4))
 
#3rd convolution layer
model.add(Conv2D(128, (4, 4),strides=(1,1), padding="same", activation='relu'))
BatchNormalization()
model.add(Conv2D(128, (4, 4),strides=(1,1),padding="same",  activation='relu'))
BatchNormalization()
model.add(MaxPooling2D(pool_size=(4,4), strides=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
BatchNormalization()
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
BatchNormalization()

model.add(Dense(num_classes, activation='softmax'))

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
 
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
 
history = model.fit_generator(train_generator, batch_size, epochs=epochs, validation_data=(x_test, y_test))

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])

#make predictions for test set
predictions = model.predict(x_test)

pred_list = []; actual_list = []
 
for i in predictions:
	pred_list.append(np.argmax(i))
 
for i in y_test:
	actual_list.append(np.argmax(i))
 
cm = confusion_matrix(actual_list, pred_list)

print(cm)

print(history.history.keys())
#model.summary()

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
