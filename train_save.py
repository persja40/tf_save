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

    print(train_data)

    train_data = np.reshape(train_data, (-1, 1))
    train_results = np.reshape(train_results, (-1, 1))
    test_data = np.reshape(test_data, (-1, 1))
    test_results = np.reshape(test_results, (-1, 1))

    print("**")
    print(train_data)

    return data, results, train_data, train_results, test_data, test_results


def generate_layers(N):
    pass

if __name__ == '__main__':
    # DUBEG CODE
    # clear screen
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
    equation = '2*x'#''2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
    lim_min = 0
    lim_max = 10
    step = 1
    # training data percentage from whole data set
    learn_pg = 0.7

    # prepare data
    data, results, train_data, train_results, test_data, test_results = \
        prepare_data(equation, lim_min, lim_max, step, learn_pg)
    # TENSORFLOW

    def model(X, w1, w2, b1, b2):
        X_w1 = tf.nn.tanh(tf.add(tf.matmul(X, w1), b1))
        y_ = tf.add(tf.matmul(X_w1, w2), b2)
        return y_

    input_size = train_data.shape[1]
    output_size = train_results.shape[1]
    h_size = 7
    learning_rate = 0.01
    training_epochs = 10

    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    y = tf.placeholder(dtype=tf.float32)

    weights_1 = tf.Variable(tf.random_normal([input_size, h_size], stddev=0.1))
    weights_2 = tf.Variable(tf.random_normal([h_size, output_size], stddev=0.1))

    bias_1 = tf.Variable(tf.random_normal([h_size]))
    bias_2 = tf.Variable(tf.random_normal([output_size]))

    y_ = model(x, weights_1, weights_2, bias_1, bias_2)
    batch_size = 100
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            epoch_loss = 0;
            i = 0
            while i < len(train_data):
                start = i
                end = i + batch_size
                batch_x = np.array(train_data[start:end])
                batch_y = np.array(train_results[start:end])
                _, loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += loss
                i += batch_size

            print("Epoch = %d,epoch loss = %.2f%%" % (epoch + 1, epoch_loss))
            correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print(y)
        print('Accuracy:', accuracy.eval({x: train_data, y: train_results}))
        output = sess.run(y_, feed_dict={x:test_data})

    print(output)
    # plt.plot(data, results, "b-")
    plt.plot(test_data, test_results, "bo")
    plt.plot(test_data, output, "go")
    plt.show()
