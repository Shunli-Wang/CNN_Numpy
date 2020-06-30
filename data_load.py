"#codeing=utf-8"

import numpy as np
import matplotlib.pyplot
import pandas as pd


# /(y.shape[1])
def one_hot(y):
    y_onehot = np.zeros((10, y.shape[1]))
    for i in range(y.shape[1]):
        y_onehot[y[0, i]][i] = 1
    return y_onehot.T


def load_data():
    data = pd.read_csv("./data/mnist_train.csv", header=None)
    train_x = data.values[:, 1:]
    trian_y = data.values[:, 0:1]
    data = pd.read_csv("./data/mnist_test.csv", header=None)
    test_x = data.values[:, 1:]
    test_y = data.values[:, 0:1]
    return train_x.T, trian_y.T, test_x.T, test_y.T


def get_dataset():
    train_x, train_y, test_x, test_y = load_data()
    train_x = train_x.T / 255.0
    test_x = test_x.T / 255.0
    train_y = one_hot(train_y)
    test_y = one_hot(test_y)
    return [train_x, test_x, train_y, test_y]


def dataset(dataset, label, bs):
    for i in range(0, dataset.shape[0] - bs + 1, bs):
        x = dataset[i:i + bs, ].reshape((bs, 1, 28, 28))
        y = label[i:i + bs, :]
        yield [x, y]
