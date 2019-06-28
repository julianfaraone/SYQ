from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from SYQ import SYQ, SYQ_Dense

from tensorflow.keras.layers import Flatten
import argparse

parser = argparse.ArgumentParser(description="SYQ MNIST - Keras")
parser.add_argument("--bit_width", default=8, type=int, help="Quantization Weight Bitwidth")


# Get dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_images.shape)
#print(len(train_labels))
#print(train_labels)
#print(test_images.shape)
#print(len(test_labels))

train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))

#Inpspect image values
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#scale
train_images = train_images / 255.0
test_images = test_images / 255.0

inputs = keras.layers.Input(shape=(28,28))

class Model:

        def __init__(self, bit_width):
                self.bit_width = bit_width

                self.model = keras.Sequential([SYQ(self.bit_width, 32, (3, 3), activation='relu', input_shape=(28,28,1))
                        , SYQ(self.bit_width, 32, (3, 3), activation='relu')
                        , Flatten()
                        , SYQ_Dense(self.bit_width, 128, activation=tf.nn.relu)
                        , SYQ_Dense(self.bit_width, 128, activation=tf.nn.relu)
                        , Dense(10, activation=tf.nn.softmax)])

        def train_model(self):
                self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

                self.model.fit(train_images, train_labels, epochs=50)

                test_loss, test_acc = model.evaluate(test_images, test_labels)

                print('Test accuracy:', test_acc)

                predictions = model.predict(test_images)

#analyze first image - this prints an array of 10 numbers
#print(predictions[0])

#most confident in
#print(np.argmax(predictions[0]))

#check if correct
#print(test_labels[0])

#predict on single image
#img = test_images[0]
# Add the image to a batch where it's the only member.
#img = (np.expand_dims(img,0))
#predictions_single = model.predict(img)
#prediction_result = np.argmax(predictions_single[0])
#print(prediction_result)

if __name__ == "__main__":
    args = parser.parse_args()

    model = Model(args.bit_width)

    model.train_model()
