import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *

# DUBEG CODE
# clear screen
import os
os.system('cls' if os.name == 'nt' else 'clear')

# END DEBUG

equation = sys.argv[1]
lim_min = int(sys.argv[2])
lim_max = int(sys.argv[3])
step = float(sys.argv[4])
learn_pg = float(sys.argv[5])

# calc data
data = np.arange(lim_min, lim_max + step, step)
results = []
for x in data:
    results.append(eval(equation))
results = np.array(results)

# shuffle data and divide to learning & testing
s = np.arange(data.shape[0])
np.random.shuffle(s)
dat = data[s]
res = results[s]

learn_max_index = floor(len(dat)*learn_pg)

l_data = dat[:learn_max_index]
l_results = res[:learn_max_index]
t_data = dat[learn_max_index:]
t_results = res[learn_max_index:]

# TENSORFLOW
# ioputs & weights
inputs = tf.placeholder(shape=[1, 1], dtype=tf.float16, name='inputs')
outputs = tf.placeholder(shape=[1, 1], dtype=tf.float16, name='outputs')
# 1st layer
hid1_size = 10
w1 = tf.Variable(tf.random_normal(
    shape=[hid1_size, 1], stddev=0.01), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=[hid1_size, 1]), name='b1')
# change tf.nn.relu to sigmoid or tanh - LATER
y1 = tf.nn.dropout(tf.nn.relu(
    tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)

# Output layer
wo = 0
bo = 0
yo = 0

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(w1))
    #print(sess.run(b1))
    print('TF')

#plt.plot(data, results, "b-")
# plt.show()
