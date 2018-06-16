import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import os
from tensorflow.python.saved_model import tag_constants


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
    equation = '2*x'#'2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
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

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'test_model')
            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('y:0')

            y_ = graph.get_tensor_by_name('add_1:0')

            output = sess.run(y_, feed_dict={x: test_data})
    plt.plot(test_data, test_results, "bo")
    plt.plot(test_data, output, "go")
    plt.show()
