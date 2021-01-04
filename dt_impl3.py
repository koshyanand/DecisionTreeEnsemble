import numpy as np
from impl3_util import load_data, preprocess_train_data, gini_function, plot, get_random_array, get_sampled_features, \
    get_leaf_prediction_value
import time
from tree import DecisionTree

train_data = 'data/pa3_train_reduced.csv'
val_data = 'data/pa3_valid_reduced.csv'


def get_feature_gain(space, feature_index, u_root):
    vals = np.empty([len(space), 2])
    vals[:, 0] = space[:, 0]
    vals[:, 1] = space[:, feature_index]
    sorted_vals = vals[np.argsort(vals[:, 1])[::1]]

    gain = 0
    threshold = 0
    prev_label = 0
    for i in range(0, len(sorted_vals)):
        # print(row.shape)
        row = sorted_vals[i, :]
        t = row[1]
        if prev_label == row[0]:
            continue
        if i != 0:
            t = (sorted_vals[i - 1, 1] + t) / 2

        val = np.split(sorted_vals, np.where(sorted_vals[:, 1] >= t)[0][:1])

        # true_space = sorted_vals[sorted_vals[:, 1] >= t]
        true_space = val[1]

        u_left = gini_function(true_space[:, 0])
        p_left = len(true_space) / len(sorted_vals)

        # false_space = sorted_vals[sorted_vals[:, 1] < t]
        false_space = val[0]

        u_right = gini_function(false_space[:, 0])
        p_right = len(false_space) / len(sorted_vals)

        gain_current = u_root - p_left * u_left - p_right * u_right
        print("u_root : ", u_root, " u_left : ", u_left, " u_right : ", u_right)
        print("GC : ", gain_current)
        if gain_current > gain:
            gain = gain_current
            threshold = t
        prev_label = row[0]
    return feature_index, gain, threshold


def create_dt_classifier(data, depth, tree, m):
    # print("Depth : ", depth)
    # print("Data Shape : ", data.shape)
    if depth == 0:
        prediction = get_leaf_prediction_value(data)
        tree.insert(None, None, prediction, True)
        return
    u_root = gini_function(data[:, 0])
    gain = 0
    feature_index = 0
    threshold = 0

    feature_sampled_data, random_indexes = get_sampled_features(data, m)
    for i in range(1, feature_sampled_data.shape[1]):
        feature_index_current, gain_current, threshold_current = get_feature_gain(feature_sampled_data, i, u_root)

        if gain_current > gain:
            gain = gain_current
            feature_index = feature_index_current
            threshold = threshold_current

    if gain == 0:
        prediction = get_leaf_prediction_value(data)
        tree.insert(None, None, prediction, True)
        return

    depth = depth - 1

    if random_indexes is not None:
        feature_index = random_indexes[feature_index]

    sorted_vals = data[np.argsort(data[:, feature_index])[::1]]

    val = np.split(sorted_vals, np.where(sorted_vals[:, feature_index] >= threshold)[0][:1])
    # true_space = data[data[:, feature_index] >= threshold]
    true_space = val[1]
    false_space = val[0]
    # false_space = data[data[:, feature_index] < threshold]
    prediction = get_leaf_prediction_value(data)
    tree.insert(threshold, feature_index, prediction, False)
    tree.left = DecisionTree()
    tree.right = DecisionTree()
    create_dt_classifier(true_space, depth, tree.left, m)
    create_dt_classifier(false_space, depth, tree.right, m)


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


def check_accuracy_with_trees(data, tree_list, depth):
    mistakes = 0
    n = len(data)
    for i in range(0, len(data)):
        output_list = []
        for tree in tree_list:
            # print(tree.print_tree("  "))
            output = get_prediction(tree, data[i, :], depth, 0)
            output_list.append(output)

        output = (max(set(output_list), key=output_list.count))
        if data[i, 0] != output:
            mistakes = mistakes + 1

    accuracy = ((n - mistakes) / n) * 100
    print(accuracy)
    return accuracy


def part1(t_data, v_data):
    tree = DecisionTree()
    tree_list = [tree]
    ts = time.time()
    create_dt_classifier(t_data, 9, tree, 0)
    print("Completed ", " Time : ", (time.time() - ts))

    train_accuracy_list = []
    val_accuracy_list = []
    iterations = []

    for i in range(0, 10):

        train_accuracy = check_accuracy_with_trees(t_data, tree_list, i)
        train_accuracy_list.append(train_accuracy)
        val_accuracy = check_accuracy_with_trees(v_data, tree_list, i)
        val_accuracy_list.append(val_accuracy)
        iterations.append(i)

    accuracy = [train_accuracy_list, val_accuracy_list]
    iters = [iterations, iterations]

    print("Completed ", " Time : ", (time.time() - ts))

    print(accuracy)
    legends = ["Training", "Validation"]
    labels = ["Accuracy in %", "Depth"]
    plot(iters, accuracy, "Accuracy Vs Depth", legends, labels)


if __name__ == '__main__':

        train_set = load_data(train_data)
        val_set = load_data(val_data)
        t_data = preprocess_train_data(train_set)
        v_data = preprocess_train_data(val_set)

        part1(t_data, v_data)

        # d = 9
        # m_list = [20]
        # n = [1, 2, 5, 10, 25]
        # train_size = len(t_data)
        # train_accuracy_list = []
        # val_accuracy_list = []
        # iterations = []
        # ts = time.time()
        #
        # for j in range(0, len(n)):
        #     print("N : ", n[j])
        #     tree_list = []
        #     for k in range(0, n[j]):
        #         randomIndexes = get_random_array(0, train_size, train_size, True)
        #         for m in m_list:
        #             tree = DecisionTree()
        #             create_dt_classifier(t_data[randomIndexes, :], d, tree, m)
        #             tree_list.append(tree)
        #
        #     train_accuracy = check_accuracy_with_trees(t_data, tree_list, d)
        #     train_accuracy_list.append(train_accuracy)
        #     val_accuracy = check_accuracy_with_trees(v_data, tree_list, d)
        #     val_accuracy_list.append(val_accuracy)
        #     iterations.append(n[j])
        #
        # accuracy = [train_accuracy_list, val_accuracy_list]
        # iters = [iterations, iterations]
        #
        # print("Completed ", " Time : ", (time.time() - ts))
        #
        # print(accuracy)
        # legends = ["Training", "Validation"]
        # labels = ["Accuracy in %", "Forests"]
        # plot(iters, accuracy, "Accuracy Vs Forests : m = 50", legends, labels)
        #


