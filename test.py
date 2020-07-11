import numpy as np
import nn


def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x.T, y


def accuracy(labels, preds):
    idxs_true = np.argmax(labels, axis=0)
    idxs_preds = np.argmax(preds, axis=0)
    return np.sum(idxs_true == idxs_preds) / idxs_true.size


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def main():
    np.random.seed(100)
    X_train, y_train = read_data('images_train.csv', 'labels_train.csv')
    y_train = one_hot_labels(y_train).T
    print(X_train.shape)
    print(y_train.shape)
    p = np.random.permutation(60000)
    X_train = X_train[:, p]
    y_train = y_train[:, p]

    X_dev = X_train[:, 0:10000]
    y_dev = y_train[:, 0:10000]
    X_train = X_train[:, 10000:]
    y_train = y_train[:, 10000:]

    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_dev = (X_dev - mean) / std

    X_test, y_test = read_data('images_test.csv', 'labels_test.csv')
    y_test = one_hot_labels(y_test).T
    X_test = (X_test - mean) / std

    model = nn.MLP(hidden_sizes=[300], hidden_activation='sigmoid', output_activation='softmax',
                   cost_function='cross_entropy')
    costs = model.fit(X_train, y_train, k=10, learning_rate=1, num_epochs=30, lmbd=0, batch_size=1000, shuffle=False)
    for item in costs:
        print(item)
    preds = model.predict(X_test)
    acc = accuracy(y_test, preds)
    print('Accuracy: {}'.format(acc))


class Foo():
    def __init__(self):
        self.x = None

    def foo(self, x):
        self.x = x

x = np.arange(10)

print(2**x)
