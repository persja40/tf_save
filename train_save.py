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

'''
equation = sys.argv[1]
lim_min = int(sys.argv[2])
lim_max = int(sys.argv[3])
step = float(sys.argv[4])
learn_pg = float(sys.argv[5])
'''
equation = '2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
lim_min = 0
lim_max = 10
step = 0.1
learn_pg = 0.7

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
def dnn_perceptron(x, weights, biases):
    for i in range(1, len(weights)):
        w = 'w'+str(i)
        b = 'b'+str(i)
        if i == 1:
            last_layer = tf.nn.sigmoid(
                tf.add(tf.matmul(x, weights[w]), biases[b]))
        else:
            last_layer = tf.nn.sigmoid(
                tf.add(tf.matmul(last_layer, weights[w]), biases[b]))
    return tf.matmul(last_layer, weights['out']) + biases['out']

# ioputs & weights
inputs = tf.placeholder(dtype=tf.float32, name='inputs', shape=[
                        None, 1])  # change to your data size
outputs = tf.placeholder(dtype=tf.float32, name='outputs', shape=[None])

# training
learning_rate = 0.1
training_epochs = 1000
cost = None
input_size = 1
hidden_layers_nr = 2
hidden_size = [10, 5]
output_size = 1

weights = {
    'w1': tf.Variable(tf.random_normal([input_size, hidden_size[0]], 0, 0.1),dtype=tf.float32),
    'w2': tf.Variable(tf.random_normal([hidden_size[0], hidden_size[1]], 0, 0.1),dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([hidden_size[1], output_size], 0, 0.1),dtype=tf.float32)
}
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_size[0]], 0, 0.1),dtype=tf.float32),
    'b2': tf.Variable(tf.random_normal([hidden_size[1]], 0, 0.1),dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([output_size], 0, 0.1),dtype=tf.float32)
}

# print(biases)
# print(weights)
dnn_perceptron([[5.0]], weights, biases)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(w1))
    # print(sess.run(b1))

#plt.plot(data, results, "b-")
# plt.show()
