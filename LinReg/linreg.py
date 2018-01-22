import tflearn
import numpy
from statistics import mean, median
import network

X = [30,40,40,50,50,50,60,70,70,80,80,80]
Y = [70,90,100,120,130,150,160,190,200,200,220,230]
#Y = [1,1,1,1,1,1,1,1,1,1,1,1]

X = numpy.array(X).reshape(12, 1)
Y = numpy.array(Y).reshape(12,1)

network = network.get_network_spec()
model = tflearn.DNN(network, tensorboard_dir='log')

model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=500, show_metric=False, run_id='openaistuff')	



	