from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
import numpy
import matplotlib.pyplot as plt

plt.figure(1, figsize=(28,28))

X, Y, testX, testY = mnist.load_data(one_hot=True)

X = numpy.split(X,4)[0]
Y = numpy.split(Y,4)[0]

X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 8, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 32, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 64, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.02,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
#model.fit({'input': X}, {'target': Y}, n_epoch=5,
#           validation_set=({'input': testX}, {'target': testY}),
#           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
#model.save('./conv.model')

model.load('./conv.model')


def predict(index):
    plt.imshow(numpy.reshape(testX[index],[28,28]), interpolation="nearest", cmap="gray")
    plt.show()
    pr = numpy.argmax(model.predict(testX[index].reshape(-1,28,28,1)))
    print("prediction ", pr)

predict(550)    
predict(1000)    
predict(18)    
predict(1200)    