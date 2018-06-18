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
    '''
    equation = sys.argv[1]
    lim_min = int(sys.argv[2])
    lim_max = int(sys.argv[3])
    step = float(sys.argv[4])
    learn_pg = float(sys.argv[5])
    '''
    # , but for now ...
    equation = '-2*sin(x)*cos(x)'#'2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
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
    learning_rate = 0.00001
    training_epochs = 1000

    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="x")
    y = tf.placeholder(dtype=tf.float32, name="y")

    y_ = model(x, [10,8,5])
    batch_size = 100
    loss = tf.reduce_mean(tf.squared_difference(y_, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate,name="optimizer").minimize(loss)

    print(y_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(training_epochs):
            #for (t,r) in zip(train_data, train_results):
            sess.run(optimizer, feed_dict={x: train_data, y: train_results})
            mse = sess.run(loss, feed_dict={x: train_data, y:train_results})
            print("Epoch = %d,MSE = %.2f" % (epoch + 1, mse))

        save_model(saver, sess)
        output = sess.run(y_, feed_dict={x:test_data})

plt.plot(test_data, test_results, "bo")
plt.plot(test_data, output, "go")
plt.show()
