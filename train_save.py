import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time

# DUBEG CODE
# clear screen
import os
os.system('cls' if os.name == 'nt' else 'clear')
# disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# END DEBUG


# params will be taken from the console
'''
equation = sys.argv[1]
lim_min = int(sys.argv[2])
lim_max = int(sys.argv[3])
step = float(sys.argv[4])
learn_pg = float(sys.argv[5])
'''
# , but for now ...
equation = '2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
lim_min = 0
lim_max = 10
step = 0.1
# training data percentage from whole data set
learn_pg = 0.7

# prepare data
data = np.arange(lim_min, lim_max + step, step)
results = []
for x in data:
    results.append(eval(equation))
results = np.array(results)

# shuffle data and divide to training & testing
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

def model(X, w1, w2):
    X_w1 = tf.nn.sigmoid(tf.matmul(X, w1))
    y_ = tf.matmul(X_w1, w2)
    return y_


l_data = np.reshape(l_data, (-1, 1))
l_results = np.reshape(l_results, (-1, 1))
t_data = np.reshape(l_data, (-1, 1))
t_results = np.reshape(l_results, (-1, 1))
print(l_data.shape)
print(l_results.shape)

input_size = l_data.shape[1]
h_size = 10
output_size = l_results.shape[1]
learning_rate = 0.01
training_epochs = 1000

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

weights_1 = tf.Variable(tf.random_normal((input_size, h_size), stddev=0.1))
weights_2 = tf.Variable(tf.random_normal((h_size, output_size), stddev=0.1))
y_ = model(X, weights_1, weights_2)

predict = tf.argmax(y_, axis=1)
#cost = tf.reduce_mean(tf.square(y_ - y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        #for i in range(len(l_data)):
        sess.run(updates, feed_dict={X: l_data, y: l_results})

        train_accuracy = np.mean(np.argmax(l_results, axis=1) ==
                                 sess.run(predict, feed_dict={X: l_data, y: l_results}))
        test_accuracy = np.mean(np.argmax(t_results, axis=1) ==
                                sess.run(predict, feed_dict={X: t_data, y: t_results}))
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
    print(sess.run(predict, feed_dict={X: l_data, y: l_results}))

#plt.plot(data, results, "b-")
# plt.show()
