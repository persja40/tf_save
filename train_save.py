import tensorflow as tf
import matplotlib.pyplot as plt
from net_utils import *
from dataset_utils import *
import os
import sys

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
    functions = eval(sys.argv[7])

    # prepare data
    data, results, train_data, train_results, test_data, test_results = \
        prepare_data(equation, lim_min, lim_max, step, learn_pg)

    input_size = train_data.shape[1]
    output_size = train_results.shape[1]
    learning_rate = 0.01
    training_epochs = 1000

    x = tf.placeholder(dtype=tf.float32, shape=[1, input_size], name="x")
    y = tf.placeholder(dtype=tf.float32, name="y")

    y_ = model(x, layers, functions)
    loss = tf.reduce_mean(tf.squared_difference(y_[0][0], y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate,name="optimizer").minimize(loss)

    performance = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(1, training_epochs+1):
            for (t,r) in zip(train_data, train_results):
                sess.run(optimizer, feed_dict={x: [t], y: [r]})

            mse = calculate_mse(sess, loss, x, y, test_data, test_results)
            performance.append(mse)
            #print one in 10 epochs
            if(epoch%10==0):
                print("Epoch = %d,MSE = %.2f" % (epoch, mse))

        save_model(saver, sess)
        #output to plot
        output = []
        for t in test_data:
            output.append(sess.run(y_, feed_dict={x:[t]})[0][0])

comparision_plot = [{
    "x_data": test_data,
    "y_data": test_results,
    "mark_type": "bo",
    "label": "Wyniki"
}, {
    "x_data": test_data,
    "y_data": output,
    "mark_type": "go",
    "label": "Odpowiedź sieci"
}]

performance_plot = [{
    "x_data": range(1, training_epochs+1),
    "y_data": performance,
    "mark_type": "-",
    "label": "MSE"
}]

plot_2d(data_list=comparision_plot, xlabel="X", ylabel="Y")
plot_2d(data_list=performance_plot, xlabel="epoka", ylabel="MSE")

plt.show()
