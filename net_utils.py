import tensorflow as tf
import numpy as np
import os
from statistics import mean

def save_model(saver, sess):
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    saver.save(sess, './saved_model/model_save')


def load_model():
    return tf.train.import_meta_graph('./saved_model/model_save.meta')


def latest_checkpoint():
    return tf.train.latest_checkpoint('./saved_model/')


def init_weights(shape, xavier_params=(None, None)):
    (fan_in, fan_out) = xavier_params
    low = -1*np.sqrt(1.0/(fan_in + fan_out))  # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(1.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def init_biases(shape):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32))


def calculate_mse(sess, loss, x, y, test_data, test_results):
    error = []
    for (t, r) in zip(test_data, test_results):
        error.append(float(sess.run(loss, feed_dict={x: [t], y: [r]})))
    return sum(error)


def model(X, num_hidden, functions):
    weights = []
    biases = []
    for layer in range(len(num_hidden)):
        if(layer == 0):
            weights.append(init_weights(
                [X.get_shape().as_list()[1], num_hidden[layer]], xavier_params=(X.get_shape().as_list()[1], num_hidden[layer])))
        else:
            weights.append(init_weights([num_hidden[layer-1], num_hidden[layer]],
                                        xavier_params=(num_hidden[layer-1], num_hidden[layer])))
        biases.append(init_biases([1, num_hidden[layer]]))
    # last layer before single output
    weights.append(init_weights([num_hidden[-1], 1], xavier_params=(num_hidden[-1], 1)))
    biases.append(init_biases([1, 1]))

    # multiply
    for layer in range(len(weights)):
        if (layer == 0):
            tmp = functions[layer](tf.add(tf.matmul(X, weights[layer]), biases[layer]))
        else:
            tmp = functions[layer](tf.add(tf.matmul(tmp, weights[layer]), biases[layer]),name='model')

    return tmp
