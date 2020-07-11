import numpy as np
from math_utils import softmax
# TODO: think of what should preds be to align with evaluation during model.fit
from utils.gradient_check import evaluate_gradient


class Loss(object):

    # returns loss, predictions, derivative
    def build(self, layer_input, true_labels, compute_derivatives=True):
        raise NotImplementedError()


class SVMMax(Loss):
    name = 'SVMOneVsAll'

    def __init__(self, margin=1.):
        self.margin = margin

    def build(self, layer_input, true_labels, compute_derivative=True):
        batch_size = layer_input.shape[0]
        preds_true_class = layer_input[range(batch_size), None, true_labels.flatten()]
        errors_ = self.margin + layer_input - preds_true_class
        errors_[range(batch_size), true_labels.flatten()] = 0.
        max_idxs = np.argmax(errors_, axis=1)
        loss_per_observation = np.maximum(errors_[range(batch_size), None, max_idxs], 0)
        loss = np.sum(loss_per_observation) / float(batch_size)
        loss_derivative = None
        if compute_derivative:
            loss_derivative = np.zeros(layer_input.shape, dtype=np.float64)
            pos_loss_mask = (loss_per_observation > 0).flatten()
            idxs_pos_loss = np.arange(batch_size)[pos_loss_mask]
            loss_derivative[idxs_pos_loss, max_idxs[pos_loss_mask]] = 1. / batch_size
            loss_derivative[idxs_pos_loss, true_labels[pos_loss_mask]] = -1. / batch_size
        return loss, np.argmax(layer_input, axis=1).reshape(layer_input.shape[0],
                                                            1), loss_derivative  # loss, predictions, derivatives


class SVM(Loss):
    name = 'SVM'

    def __init__(self, margin=1.):
        self.margin = margin

    def build(self, layer_input, true_labels, compute_derivative=True):
        batch_size = layer_input.shape[0]
        preds_true_class = layer_input[range(batch_size), None, true_labels.flatten()]
        errors_ = self.margin - preds_true_class + layer_input
        errors_[range(batch_size), true_labels.flatten()] = 0.
        loss_per_observation = np.sum(np.maximum(errors_, 0), keepdims=True, axis=1)
        loss = np.sum(loss_per_observation) / float(batch_size)
        loss_derivative = None
        if compute_derivative:
            dy = (errors_ > 0).astype(float)
            loss_derivative = dy
            loss_derivative[range(batch_size), true_labels.flatten()] = - np.sum(dy, axis=1)
            loss_derivative /= float(batch_size)
        return loss, np.argmax(layer_input, axis=1).reshape(layer_input.shape[0], 1), loss_derivative


class CrossEntropyLoss(Loss):
    name = 'CrossEntropy'

    def build(self, layer_input, true_labels, compute_derivative=True):
        batch_size = layer_input.shape[0]
        probs_unnorm, Z, shifted_input = softmax(layer_input, return_separately=True)
        log_probs = shifted_input - np.log(Z)
        loss = -np.sum(log_probs[range(batch_size), true_labels.flatten()]) / float(batch_size)
        loss_derivative = None
        predicted_labels = np.argmax(log_probs, axis=1).reshape(log_probs.shape[0], 1)
        if compute_derivative:
            loss_derivative = probs_unnorm / Z
            loss_derivative[range(batch_size), true_labels.flatten()] += -1.
            loss_derivative /= float(batch_size)
        return loss, predicted_labels, loss_derivative


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


if __name__ == '__main__':
    from math_utils import relative_error
    import time

    c = 10
    batch_size = 5
    shape = (batch_size, c)
    X = (np.random.random_sample(shape[0] * shape[1]) * 10 - 5).reshape(shape)
    y = np.random.randint(c, size=batch_size)

    for loss in [CrossEntropyLoss(), SVM(), SVMMax()]:
        f = lambda x: loss.build(x, y, compute_derivative=False)[0]
        t0 = time.time()

        l0, _, g1 = loss.build(X, y)
        print()
        print()
        print(type(loss))
        print('Execution time: {}'.format(time.time() - t0))
        g2 = evaluate_gradient(f, X)
        err = relative_error(g1, g2)
        assert err[0] < 0.0001
        print(err[0], g1[err[1]], g2[err[1]])
        b1 = isinstance(loss, CrossEntropyLoss)
        b2 = isinstance(loss, SVM)
        if b1 or b2:
            loss2 = softmax_loss if b1 else svm_loss
            l1, g3 = loss2(X, y)
            err, idx = relative_error(g1, g3)
            print('Losses: my {}, cs231n {}'.format(l0, l1))
            print('My vs cs231 grads: {}, {}, {}'.format(err, g1[idx], g3[idx]))
            err, idx = relative_error(g2, g3)
            print('Num vs cs231 grads: {}, {}, {}'.format(err, g2[idx], g3[idx]))
