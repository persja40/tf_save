import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from net_utils import *
from dataset_utils import *
import sys
import os

if __name__ == '__main__':
    # os.system('cls' if os.name == 'nt' else 'clear')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # params will be taken from the console
    equation = sys.argv[1]
    lim_min = int(sys.argv[2])
    lim_max = int(sys.argv[3])
    step = float(sys.argv[4])
    learn_pg = float(sys.argv[5])
    layers = eval(sys.argv[6])
    directory = str(sys.argv[7])


    print(directory)
    print(type(directory))

    # prepare data
    data, results, train_data, train_results, test_data, test_results = \
        prepare_data(equation, lim_min, lim_max, step, learn_pg)

    input_size = train_data.shape[1]
    output_size = train_results.shape[1]
    learning_rate = 0.01
    training_epochs = 100

    x = tf.placeholder(dtype=tf.float32, shape=[1, input_size], name="x")
    y = tf.placeholder(dtype=tf.float32, name="y")


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.restore(sess, directory)
        model = load_model()
        output = []
        for t in test_data:
            output.append(sess.run(model, feed_dict={x:[t]})[0][0])

plt.plot(test_data, output, "go")
plt.show()
