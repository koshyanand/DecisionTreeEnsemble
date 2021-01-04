import numpy as np
from impl3_util import load_data, preprocess_train_data, plot, ada_gini_function, ada_get_leaf_prediction_value, \
    get_prediction
import time
from tree import DecisionTree

train_data = 'data/pa3_train_reduced.csv'
val_data = 'data/pa3_valid_reduced.csv'


def get_feature_gain(space, feature_index, u_root):
    # print("Feature : ", feature_index)
    vals = np.empty([len(space), 3])
    vals[:, 0] = space[:, 0]
    vals[:, 1] = space[:, 1]
    vals[:, 2] = space[:, feature_index]
    sorted_vals = vals[np.argsort(vals[:, 2])[::1]]

    gain = 0
    threshold = 0
    prev_label = 0
    sum_sorted = np.sum(sorted_vals[:, 1])

    for i in range(0, len(sorted_vals)):
        # print(row.shape)
        row = sorted_vals[i, :]
        t = row[2]
        # print(t)
        if prev_label == row[0]:
            continue
        if i != 0:
            t = (sorted_vals[i - 1, 2] + t) / 2

        val = np.split(sorted_vals, np.where(sorted_vals[:, 2] >= t)[0][:1])

        # true_space = sorted_vals[sorted_vals[:, 1] >= t]
        true_space = val[1]
        # print(true_space)
        u_left = ada_gini_function(true_space[:, 0:2])
        p_left = (np.sum(true_space[:, 1]) * len(true_space)) / (sum_sorted * len(sorted_vals))
        # print("u_left : ", u_left, " p_left : ", p_left)
        # print("True Space : ", true_space.shape)
        # false_space = sorted_vals[sorted_vals[:, 1] < t]
        false_space = val[0]

        u_right = ada_gini_function(false_space[:, 0:2])
        p_right = (np.sum(false_space[:, 1]) * len(false_space)) / (sum_sorted * len(sorted_vals))
        # print("u_right : ", u_right, " p_right : ", p_right)
        # print("False Space : ", false_space.shape)

        gain_current = u_root - p_left * u_left - p_right * u_right
        # gain_current = u_root - u_left - u_right
        # print("u_root : ", u_root, " u_left : ", u_left, " u_right : ", u_right)
        # print("p_left : ", p_left, " p_right : ", p_right)
        # print("FG : ", gain_current, " Threshold : ", t)
        if gain_current > gain:
            gain = gain_current
            threshold = t
        prev_label = row[0]
    return feature_index, gain, threshold


def create_dt_classifier(data, depth, tree):
    # print(data.shape, " Depth : ", depth)
    if depth == 0:
        prediction = ada_get_leaf_prediction_value(data)
        tree.insert(None, None, prediction, True)
        return
    u_root = ada_gini_function(data[:, 0:2])
    # print(u_root)
    gain = 0
    feature_index = 0
    threshold = 0

    for i in range(2, data.shape[1]):
        feature_index_current, gain_current, threshold_current = get_feature_gain(data, i, u_root)
        # print("gain_current : ", gain_current, " threshold_current : ", threshold, " feature_index_current", feature_index_current)
        if gain_current > gain:
            gain = gain_current
            feature_index = feature_index_current
            threshold = threshold_current
        # break
        # print("gain : ", gain, " threshold : ", threshold, " feature_index", feature_index)

    if gain == 0:
        prediction = ada_get_leaf_prediction_value(data)
        tree.insert(None, None, prediction, True)
        return

    depth = depth - 1

    sorted_vals = data[np.argsort(data[:, feature_index])[::1]]

    val = np.split(sorted_vals, np.where(sorted_vals[:, feature_index] >= threshold)[0][:1])
    # true_space = data[data[:, feature_index] >= threshold]
    true_space = val[1]
    false_space = val[0]
    # false_space = data[data[:, feature_index] < threshold]
    prediction = ada_get_leaf_prediction_value(data)
    tree.insert(threshold, feature_index, prediction, False)
    tree.left = DecisionTree()
    tree.right = DecisionTree()
    create_dt_classifier(true_space, depth, tree.left)
    create_dt_classifier(false_space, depth, tree.right)


def check_accuracy_with_trees(data, tree_list, depth, alpha_list):
    mistakes = 0
    n = len(data)

    for i in range(0, len(data)):
        output_sum = 0
        for j in range(0, len(tree_list)):
            tree = tree_list[j]
            alpha = alpha_list[j]
            # print(tree.print_tree("  "))
            output = get_prediction(tree, data[i, :], depth, 0)
            output_sum = output_sum + output * alpha

        if np.sign(output_sum) != data[i, 0]:
            mistakes = mistakes + 1

    accuracy = ((n - mistakes) / n) * 100
    print(accuracy)
    return accuracy


def get_params(data, tree, depth):
    output_list = []
    indicator_list = []
    error = 0
    for i in range(0, len(data)):
        output = get_prediction(tree, data[i, :], depth, 0)
        output_list.append(output)

        if data[i, 0] != output:
            indicator = 1
            indicator_alt = 1
        else:
            indicator = 0
            indicator_alt = -1
        indicator_list.append(indicator_alt)
        error = error + data[i, 1] * indicator
    error = error / np.sum(data[:, 1])
    alpha = 0.5 * np.log((1 - error) / error)
    data[:, 1] = np.multiply(data[:, 1], np.exp(alpha * np.array(indicator_list)))
    return alpha


if __name__ == '__main__':

    train_set = load_data(train_data)
    val_set = load_data(val_data)
    t_data = preprocess_train_data(train_set)
    v_data = preprocess_train_data(val_set)

    d = 9
    l_list = [1, 5, 10, 20]
    train_size = len(t_data)

    train_accuracy_list = []
    val_accuracy_list = []
    iterations = []
    ts = time.time()

    D = np.full((len(t_data)), 1/len(t_data))
    print(D)
    ada_test_data = np.insert(t_data, 1, D, axis=1)
    ada_val_data = np.insert(v_data, 1, np.full(len(v_data), 1), axis=1)

    # print(ada_test_data)
    for j in range(0, len(l_list)):
        print("L : ", l_list[j])
        tree_list = []
        alpha_list = []
        ada_test_data = np.insert(t_data, 1, D, axis=1)
        # np.zeros((2, 1))
        for l in range(0, l_list[j]):
            tree = DecisionTree()
            create_dt_classifier(ada_test_data, d, tree)
            tree_list.append(tree)
            alpha = get_params(ada_test_data, tree, d)
            alpha_list.append(alpha)
            print(ada_test_data[:, 1])
            # print(ada_test_data)

        train_accuracy = check_accuracy_with_trees(ada_test_data, tree_list, d, alpha_list)
        train_accuracy_list.append(train_accuracy)
        val_accuracy = check_accuracy_with_trees(ada_val_data, tree_list, d, alpha_list)
        val_accuracy_list.append(val_accuracy)
        iterations.append(l_list[j])
        # print(train_accuracy)
        # print(val_accuracy)
    print(train_accuracy_list)
    print(val_accuracy_list)
    accuracy = [train_accuracy_list, val_accuracy_list]
    iters = [iterations, iterations]

    print("Completed ", " Time : ", (time.time() - ts))

    print(accuracy)
    legends = ["Training", "Validation"]
    labels = ["Accuracy in %", "L"]
    plot(iters, accuracy, "Accuracy Vs L", legends, labels)



