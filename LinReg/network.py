import tflearn
 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3

def get_network_spec():
	network = input_data(shape=[None, 1], name='input')

	network = fully_connected(network, 64, activation='relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 64, activation='relu')
	network = dropout(network, 0.8)	
	
	network = fully_connected(network, 1, activation='softmax')
	
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
	
	return network