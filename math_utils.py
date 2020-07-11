import numpy as np


def sigmoid(z):
    neg_mask = (z < 0)
    pos_mask = np.logical_not(neg_mask)

    x = np.zeros_like(z, dtype=float)
    x[pos_mask] = np.exp(-z[pos_mask])
    x[neg_mask] = np.exp(z[neg_mask])

    numerator = np.ones_like(z, dtype=float)
    numerator[neg_mask] = x[neg_mask]
    res = numerator / (1 + x)
    return res


def sigmoid_derivative(z):
    f = sigmoid(z)
    return f * (1 - f)


def softmax(z, return_separately=False):
    """

    :param z: numpy.array, input
    :param return_separately: bool, if True the output will be
        (unnormalized_probs, normalization_coeff, shifted_input), else - probs
    :return: z turned into distribution
    """
    max_idxs = np.argmax(z, axis=1)
    res = z - z[range(z.shape[0]), None, max_idxs]
    exp_z = np.exp(res)
    if return_separately:
        return exp_z, np.sum(exp_z, axis=1, keepdims=True), res
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_derivative(z):
    f = softmax(z)
    return f * (1 - f)


def relative_error(x, y):
    error = np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y)))
    return np.max(error), np.unravel_index(np.argmax(error), shape=error.shape)
