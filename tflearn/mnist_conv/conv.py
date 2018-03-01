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
import cv2

network_size = 8

plt.figure(1, figsize=(28,28))

X, Y, testX, testY = mnist.load_data(one_hot=True)

X = numpy.split(X,4)[0]
Y = numpy.split(Y,4)[0]

X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])


# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, network_size*1, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, network_size*2, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, network_size*4, activation='tanh', regularizer="L2")
network = dropout(network, 0.8)
network = fully_connected(network, network_size*8, activation='tanh', regularizer="L2")
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.02,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)

# Training
#model.fit({'input': X}, {'target': Y}, n_epoch=5, validation_set=({'input': testX}, {'target': testY}), snapshot_step=100, show_metric=True, run_id='convnet_mnist')
#model.save('./conv.model')

model.load('./conv.model')


def visual_test(arr):
    plt.imshow(numpy.reshape(arr,[28,28]), interpolation="nearest", cmap="gray")
    plt.show()    
    pr = predict(arr)
    print("prediction ", pr)

def predict(arr):
    return numpy.argmax(model.predict(arr.reshape(-1,28,28,1)))    

#visual_test(testX[550])    
#visual_test(testX[1000])    
# visual_test(18)    
# visual_test(1200)    

def eval_network():
    results = []
    for i in range(0,100):
        x = testX[i]
        y = numpy.argmax(testY[i])
        hy = predict(x)
        results.append(y==hy)
        if (y!=hy):
            print('wrong prediction at index: ',i)
    acc = sum(results)/len(results)
    print('prediction accuracy: ', acc*100)
    
#eval_network()

im = cv2.imread("examples/ex_2_2.png")   
img = [0]*784
img = numpy.reshape(img, [28,28])
for y in range(0,27):
    for x in range(0,27):
        img[y][x] = im[y][x][0]
visual_test(img)
