import tensorflow as tf
import numpy as np

A = [1,1,1,0,1]
B = [1,1,1,0,0]
TFP = tf.cast( tf.equal(A,B), tf.int32)
TP = tf.cast(tf.equal(TFP,B),tf.int32)

with tf.Session() as sess:
    print(
        sess.run(
            TP
        )
    )