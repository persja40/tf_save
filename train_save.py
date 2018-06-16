import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import os

def shuffle_dataset(data, results):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    shuffled_data = data[s]
    shuffled_results = results[s]
    return shuffled_data, shuffled_results


def divide_dataset(learn_max_index, shuffled_data):
    train_data = shuffled_data[:learn_max_index]
    test_data = shuffled_data[learn_max_index:]

    return train_data, test_data


def prepare_data(equation, lim_min, lim_max, step, learn_pg):
    data = np.arange(lim_min, lim_max + step, step)
    results = np.array([eval(equation) for x in data])

    shuffled_data, shuffled_results = shuffle_dataset(data, results)
    learn_max_index = floor(len(shuffled_data) * learn_pg)
    train_data, test_data = divide_dataset(learn_max_index, shuffled_data)
    train_results, test_results = divide_dataset(learn_max_index, shuffled_results)

    train_data = np.reshape(train_data, (-1, 1))
    train_results = np.reshape(train_results, (-1, 1))
    test_data = np.reshape(test_data, (-1, 1))
    test_results = np.reshape(test_results, (-1, 1))
    return data, results, train_data, train_results, test_data, test_results


def init_weights(shape, xavier_params = (None, None)):
    (fan_in, fan_out) = xavier_params
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32))

def model(X, num_hidden=10):
    w_h = init_weights([1, num_hidden], xavier_params=(1, num_hidden))
    b_h = init_biases([1, num_hidden])
    h = tf.nn.tanh(tf.matmul(X, w_h) + b_h)

    w_o = init_weights([num_hidden, 1], xavier_params=(num_hidden, 1))
    b_o = init_biases([1, 1])
    return tf.matmul(h, w_o) + b_o


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    step = 0.01
    # training data percentage from whole data set
    learn_pg = 0.7

    # prepare data
    data, results, train_data, train_results, test_data, test_results = \
        prepare_data(equation, lim_min, lim_max, step, learn_pg)

    input_size = train_data.shape[1]
    output_size = train_results.shape[1]
    h_size = 40
    learning_rate = 0.01
    training_epochs = 1000

    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    y = tf.placeholder(dtype=tf.float32)

    y_ = model(x, h_size)
    batch_size = 100
    optimizer = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(y_ - y))
    mse = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            i = 0
            for i in range(training_epochs):
                start = i
                end = i + batch_size
                batch_x = np.array(train_data[start:end])
                batch_y = np.array(train_results[start:end])
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                i += batch_size
            mse = sess.run(tf.nn.l2_loss(y_ - train_results), feed_dict={x: train_data})
            print("Epoch = %d,MSE = %.2f" % (epoch + 1, mse))
            if mse < 0.01:
                break

        output = sess.run(y_, feed_dict={x:test_data})

    plt.plot(test_data, test_results, "bo")
    plt.plot(test_data, output, "go")
    plt.show()
