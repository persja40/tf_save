import tensorflow as tf
import numpy as np
import os


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
    low = -1*np.sqrt(6.0/(fan_in + fan_out))  # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def init_biases(shape):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32))


def model(X, num_hidden):
    weights = []
    biases = []
    for layer in range(len(num_hidden)):
        if(layer == 0):
            print(X.shape[1])
            print(num_hidden[layer])
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
            tmp = tf.add(tf.matmul(X, weights[layer]), biases[layer])
        else:
            tmp = tf.add(tf.matmul(tmp, weights[layer]), biases[layer])

    return tf.nn.sigmoid(tmp,name='model')
