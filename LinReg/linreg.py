import tflearn
import numpy
from statistics import mean, median
import network

X = [30,40,40,50,50,50,60,70,70,80,80,80]
Y = [70,90,100,120,130,150,160,190,200,200,220,230]


network = network.get_network_spec()
model = tflearn.DNN(network, tensorboard_dir='log')

model.fit(X, Y, n_epoch=1000, show_metric=True)	

prediction = model.predict([30,40,50,60,70,80])
print("\n\nPredticion")
print(prediction)


	