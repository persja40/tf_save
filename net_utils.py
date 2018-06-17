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
    h = tf.nn.tanh(tf.add(tf.matmul(X, w_h), b_h))

    w_o = init_weights([num_hidden, 1], xavier_params=(num_hidden, 1))
    b_o = init_biases([1, 1])
    return tf.add(tf.matmul(h, w_o), b_o, name="model")
