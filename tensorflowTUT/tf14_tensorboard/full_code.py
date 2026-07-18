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


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.compat.v1.Variable(tf.compat.v1.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.compat.v1.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=1))

with tf.name_scope('train'):
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.compat.v1.Session()

writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs


