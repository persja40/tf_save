import numpy as np
from math import *
import matplotlib.pyplot as plt

def shuffle_dataset(data, results):
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

def plot_2d(data_list=[], xlabel="", ylabel=""):
    plt.figure()
    for data in data_list:
        plt.plot(data["x_data"], data["y_data"], data["mark_type"], label=data["label"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()