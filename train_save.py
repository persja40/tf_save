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
#disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# END DEBUG



# params will be taken from the console
'''
equation = sys.argv[1]
lim_min = int(sys.argv[2])
lim_max = int(sys.argv[3])
step = float(sys.argv[4])
learn_pg = float(sys.argv[5])
'''
#, but for now ...
equation = '2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
lim_min = 0
lim_max = 10
step = 0.1
#training data percentage from whole data set
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
# output calc, multiplying matrices
def dnn_perceptron(x, weights, biases):
    for i in range(1, len(weights)):
        w = 'w'+str(i)
        b = 'b'+str(i)
        if i == 1:
            # if x.get_shape()[0] == 1:
            #     last_layer = tf.nn.sigmoid(tf.add(tf.scalar_mul(x[0], weights[w]), biases[b]))
            # else:
                last_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[w]), biases[b]))
        else:
            last_layer = tf.nn.sigmoid(
                tf.add(tf.matmul(last_layer, weights[w]), biases[b]))
    return tf.matmul(last_layer, weights['out']) + biases['out']


# training params
learning_rate = e-4
training_epochs = 1000
cost = None
input_size = 1
hidden_layers_nr = 2
hidden_size = [10, 5]
output_size = 1
# debug print one in print_step
print_step = 100

# ioputs & weights
inputs = tf.placeholder(dtype=tf.float32, name='inputs', shape=[1, input_size])
outputs = tf.placeholder(dtype=tf.float32, name='outputs', shape=[1, output_size])

weights = {
    'w1': tf.Variable(tf.random_normal([input_size, hidden_size[0]], 0, 0.1), dtype=tf.float32),
    'w2': tf.Variable(tf.random_normal([hidden_size[0], hidden_size[1]], 0, 0.1), dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([hidden_size[1], output_size], 0, 0.1), dtype=tf.float32)
}
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_size[0]], 0, 0.1), dtype=tf.float32),
    'b2': tf.Variable(tf.random_normal([hidden_size[1]], 0, 0.1), dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([output_size], 0, 0.1), dtype=tf.float32)
}

model = dnn_perceptron(inputs, weights, biases)
cost = tf.reduce_mean(tf.square(model - outputs))  # mse
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

start = time.time()
# detailed log
# config=tf.ConfigProto(log_device_placement=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for l, r in zip(l_data, l_results):
            #print("\n LEARN: {}\n".format(learn))
            _, c, p = sess.run([optimizer, cost, model], feed_dict={
                               inputs: np.reshape( l, (1, input_size) ),
                               outputs: np.reshape( r, (1, output_size) )})

        if epoch % print_step == 0:
            print('Learning epoch: {}'.format(epoch))
            print('MSE: '.format())
            print('')

    # check results
    i = 0
    test = []
    for x in data:
        print(x)
        test.append( sess.run(model, feed_dict={inputs: np.reshape( x, (1, input_size) )})[0][0] )
        i+= 1

print(test)

plt.plot(data, results, "b-")
plt.plot(data, test, "r-")
plt.show()
end = time.time()
print("\n\n\n Czas: {}".format(end-start))