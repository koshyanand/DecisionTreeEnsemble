import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


def load_data(link):
    my_data = genfromtxt(link, delimiter=',')
    return my_data


def output_function(Y):
    if Y == 3:
        return 1
    else:
        return -1

def preprocess_train_data(data):
    Y = data[:, 0]
    vecfunc = np.vectorize(output_function)
    Y = vecfunc(Y)
    # print(Y)
    data[:, 0] = Y
    # X = data
    return data


def gini_function(v):
    n = len(v)
    if n == 0:
        return 0

    unique, counts = np.unique(v, return_counts=True)
    # print("Unique : ", unique, " Counts : ", counts)
    if len(unique) == 1:
        return 0
    if unique[0] == -1:
        p_neg = counts[0] / n
        p_pos = counts[1] / n
    else:
        p_neg = counts[1] / n
        p_pos = counts[0] / n
    # print("Gini : ", p_neg, " ", p_pos)
    entropy = 1 - p_neg ** 2 - p_pos ** 2
    return entropy


def plot(iterations, costLists, title, legends, labels):
    # fig, ax = plt.subplots()
    colorsList = ['blue', 'red', 'green', 'yellow']
    print(legends)
    for i in range(0, len(iterations)):
        c = colorsList[i]
        iteration = iterations[i]
        costList = costLists[i]
        plt.plot(iteration, costList, c=c, label=legends[i], markeredgewidth=2)

    plt.ylabel(labels[0])
    plt.xlabel(labels[1])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()


def get_random_array(min, max, size, is_with_replacement):
    # print(min, " : ", max, " : ", size)
    return np.random.choice(range(min, max), size, replace=is_with_replacement)


def get_sampled_features(data, m):
    if m == 0:
        return data, None
    random_indexes = get_random_array(1, data.shape[1], m, False)
    random_indexes = np.insert(random_indexes, 0, 0, axis=0)
    feature_sampled_data = data[:, random_indexes]
    return feature_sampled_data, random_indexes


def get_leaf_prediction_value(data):
    vals = data[:, 0]
    n = len(vals)
    # print("n : ", n)
    unique, counts = np.unique(vals, return_counts=True)
    if len(unique) == 1:
        if unique[0] == -1:
            return -1
        else:
            return 1
    if unique[0] == -1:
        p_neg = counts[0] / n
        p_pos = counts[1] / n
    else:
        p_neg = counts[1] / n
        p_pos = counts[0] / n

    if p_pos > p_neg:
        return 1
    else:
        return -1


def ada_gini_function(v):
    n = len(v)
    if n == 0:
        return 0
    sorted_vals = v[np.argsort(v[:, 0])[::1]]
    val = np.split(sorted_vals, np.where(sorted_vals[:, 0] > 0)[0][:1])
    if len(val[0]) == 0 or len(val) == 1:
        return 0
    p_neg = 0
    p_pos = 0
    if val[0][0, 0] == -1:
        sum_sorted_weights = np.sum(sorted_vals[:, 1]) * n
        p_neg = (np.sum(val[0][:, 1]) * len(val[0])) / sum_sorted_weights
        p_pos = (np.sum(val[1][:, 1]) * len(val[1])) / sum_sorted_weights

    # print("P_neg : ", p_neg, " P_pos : ", p_pos)
    entropy = 1 - p_neg ** 2 - p_pos ** 2
    return entropy


def ada_get_leaf_prediction_value(data):
    vals = data[:, 0:2]
    n = len(vals)
    # print("n : ", n)

    sorted_vals = vals[np.argsort(vals[:, 0])[::1]]
    sorted_weights_sum = np.sum(sorted_vals[:, 1])
    p_pos = 0
    p_neg = 0
    val = np.split(sorted_vals, np.where(sorted_vals[:, 0] > 0)[0][:1])
    if len(val[0]) > 0 and val[0][0, 0] == -1:
        p_neg = np.sum(val[0][:, 1]) * len(val[0]) / (n * sorted_weights_sum)
        if len(val) > 1:
            p_pos = np.sum(val[1][:, 1]) * len(val[1]) / (n * sorted_weights_sum)
    else:
        p_pos = (np.sum(val[1][:, 1]) * len(val[1])) / (n * sorted_weights_sum)
        # if len(val) > 1:
        #     p_neg = np.sum(val[1][:, 1])

    if p_pos > p_neg:
        return 1
    else:
        return -1


def take_second(elem):
    return elem[1]


def get_prediction(tree, e, depth, depth_count):
    if tree.is_leaf:
        return tree.prediction

    if depth == depth_count:
        return tree.prediction

    depth_count = depth_count + 1
    if e[tree.feature].reshape(1, 1) >= tree.threshold:
        return get_prediction(tree.left, e, depth, depth_count)
    else:
        return get_prediction(tree.right, e, depth, depth_count)