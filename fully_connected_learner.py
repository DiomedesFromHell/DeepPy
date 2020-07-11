import pickle
import numpy as np
import os
from six.moves import cPickle as pickle
from layers import Affine, ReLuActivation, ConvolutionNaive, Flatten
from losses import CrossEntropyLoss, SVMMax, SVM
from initializers import UniformInitializer, HeInitializer, NormalInitializer
from model import Model
from model import save_callback
from model import accuracy_metric
from optimizers import GradientDescent, SGDMomentum, SGDNesterovMomentum, RMSProp, Adam


def load_pickle(f):
    return pickle.load(f, encoding="latin1")


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ds_root):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ds_root, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ds_root, "test_batch"))
    return Xtr, Ytr, Xte, Yte


if __name__ == '__main__':
    X_tr, y_tr, X_te, y_te = load_CIFAR10('/home/vaszac/PycharmProjects/cs231/assignment2/cifar-10-batches-py')
    print(X_tr.shape)
    # X_tr = X_tr.reshape((X_tr.shape[0], -1))
    # X_te = X_te.reshape((X_te.shape[0], -1))
    # print(np.amax(X_tr))
    X_tr /= 255.
    X_te /= 255.
    X_tr = X_tr[:50, :, :, :]
    y_tr = y_tr[:50]
    mean = np.mean(X_tr, axis=0)
    X_tr -= mean
    X_te -= mean
    y_tr = y_tr.reshape((-1, 1))
    y_te = y_te.reshape((-1, 1))
    # X_tr = X_tr[:50, :]
    # y_tr = y_tr[:50, :]
    model = Model(verbose=True)
    batch_size = 25
    n_classes = 10
    std = 0.01
    reg = 0.0

    model.add_layer(ConvolutionNaive(32, (3, 3), input_shape=(batch_size, X_tr.shape[1], X_tr.shape[2], X_tr.shape[3]),
                                     weight_initializer=NormalInitializer(std)))
    model.add_layer(ReLuActivation())
    model.add_layer(ConvolutionNaive(32, (3, 3), weight_initializer=NormalInitializer(std)))
    model.add_layer(Flatten())
    model.add_layer(ReLuActivation())

    model.add_layer(Affine(100, weight_initializer=NormalInitializer(std), reg=reg))
    model.add_layer(ReLuActivation())
    model.add_layer(Affine(n_classes, weight_initializer=NormalInitializer(std), reg=reg))

    model.initialize(loss=CrossEntropyLoss(),
                     optimizer=Adam(learning_rate=0.001, decay_fst_mom=0.9, decay_sec_mom=0.999))
    # with open('model_90_49.14262959724404', 'rb') as file:
    #     model = pickle.load(file)
    model.fit(batch_size, X_tr, y_tr, n_epochs=100, metric=accuracy_metric)
