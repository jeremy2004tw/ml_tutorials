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

state = tf.compat.v1.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.compat.v1.assign(state, new_value)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

