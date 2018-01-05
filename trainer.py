from __future__ import print_function
import tensorflow as tf



a = tf.Variable([100.], dtype=tf.float32)
b = tf.Variable([100.], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = a*x + b

init = tf.global_variables_initializer()

sess = tf.Session();
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [1, 2, 3, 4]}))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) 
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [1, 2, 3, 4]})

print(sess.run([a, b]))