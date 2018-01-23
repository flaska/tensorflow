import tflearn

def get_network_spec():
	network = tflearn.input_data(shape=[None])
	network = tflearn.single_unit(network)
	network = tflearn.regression(network, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)	
	return network


def get_network_spec():
	network = tflearn.input_data(shape=[None])
	network = tflearn.single_unit(network)
	network = tflearn.regression(network, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)	
	return network