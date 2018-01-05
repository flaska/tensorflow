from __future__ import print_function
import tensorflow as tf


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

sum = a + b

sess = tf.Session();
print("sum:", sum)
print("sess.run(sum):", sess.run(sum, {a: 3, b: 4.5}))
