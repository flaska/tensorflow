
import tensorflow as tf
import tflearn

graph = tf.Graph()
with graph.as_default():
    net = tflearn.input_data([None, 2])
    net = tflearn.fully_connected(net,6,
                activation='tanh',weights_init='normal')

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('tmp/tensorboard_log', sess.graph)