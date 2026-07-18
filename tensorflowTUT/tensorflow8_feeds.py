# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

# Enable TensorFlow 1.x-style graph execution for this tutorial example
tf.compat.v1.disable_eager_execution()

input1 = tf.compat.v1.placeholder(tf.float32)
input2 = tf.compat.v1.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.compat.v1.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
