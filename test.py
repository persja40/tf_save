import tensorflow as tf
from dataset_utils import *
from net_utils import *
import matplotlib.pyplot as plt
import os
from tensorflow.python.saved_model import tag_constants


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
    equation = '2*sin(x)*cos(x)'#'2*sin(1.5*x-5)*cos(-3.2*x+1.7)'
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

    with tf.Session() as sess:
        imported_meta = load_model()
        imported_meta.restore(sess, latest_checkpoint())
        output = sess.run('model:0', feed_dict={'x:0': test_data})
    plt.plot(test_data, test_results, "bo")
    plt.plot(test_data, output, "go")
    plt.show()
