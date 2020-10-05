import os
import pickle
import numpy as np
from math_utils import softmax
from optimizers import GradientDescent


class Model(object):

    def __init__(self, batch_size=32, layers=[], optimizer=GradientDescent(), verbose=False, loss=None, seed=None):
        """

        :param batch_size: batch size, opt
        :param layers: layers of sequential model, opt
        :param optimizer: optimization algorithm, defines update rules for weights, opt
        :param verbose: if True, print training info to console
        :param loss: loss function
        """
        self.batch_size = batch_size
        self.layers = layers
        self.n_layers = len(layers)
        self.verbose = verbose
        self.loss = loss
        self.size = 0  # be careful, size can be correctly set only after initialize() call
        self.optimizer = optimizer
        self.seed = seed

    def add_layer(self, layer):
        """

        Push new layer to the top of layers sequence.
        Note, that the first layer should have input shape specified.

        :param layer: layer

        """
        self.layers.append(layer)
        self.n_layers += 1

    def _get_weight_indices(self, layer_number):
        """

        Get start and end indices for weights of the given layer.
        Needed for optimization algorithm to update weights.

        :param layer_number: layer order
        :return: start and end indices.
        """
        left_idx = 0
        for i in range(layer_number):
            left_idx += self.layers[i].size
        right_idx = left_idx + self.layers[layer_number].size
        return left_idx, right_idx

    def initialize(self, optimizer=None, loss=None):
        """

        Initialize layer weights and their shape.
        Set size to optimizer, it is needed if optimizer uses
        history of training for each weight.

        :param optimizer: optimizer, opt
        :param loss: loss function, opt
        """
        if loss:
            self.loss = loss

        if self.layers and not self.layers[0].input_shape[1]:
            raise ValueError('Model.add_layer: first layer should have shape specified.')

        for i in range(1, len(self.layers)):
            self.layers[i].input_shape = self.layers[i - 1].output_shape

        for i, layer in enumerate(self.layers):
            layer.initialize_parameters()
            layer.name = '{}_{}'.format(layer.layer_type, i)
            self.size += layer.size

        if optimizer:
            self.optimizer = optimizer
            self.optimizer.size = self.size

    def fit(self, batch_size, X_train, y_train, X_val=None, y_val=None, n_epochs=10, shuffle=True,
            on_epoch_callbacks=[], metric=None):
        """

        Train model.
        Note that layer weights should be initialized
            before calling fit as well as loss and optimizer should be set.

        :param batch_size: batch size
        :param X_train: observations of the shape (n_observations, d1, d2, d3, ...), where
            di is size of the i-th dimension of input feature vector. Used for training
        :param y_train: training labels
        :param X_val: validation observations
        :param y_val: validation labels
        :param n_epochs: total number of epochs
        :param shuffle: if True, shuffle training data before new epoch.
        :param on_epoch_callbacks: functions that are called at the end of the epoch.
        :param metric: metric for evaluation of the model. It is computed at the end
            of an each iteration for training batch and at the end of an each epoch for X_val.
        :return: training log. Has keys: train_loss, train_metric, val_loss, val_metric.
        """
        n_observations = X_train.shape[0]
        n_train_batches, last_train_batch_size = n_observations // batch_size, n_observations % batch_size

        if last_train_batch_size > 0:
            n_train_batches += 1

        train_idxs = np.arange(n_observations)

        train_log = {'train_loss': [], 'val_loss': []}

        if metric:
            train_log['train_metric'], train_log['val_metric'] = [], []

        iteration = 0
        for epoch in range(n_epochs):
            train_log['current_epoch'] = epoch
            if shuffle:
                np.random.shuffle(train_idxs)

            if self.verbose:
                print('\nEpoch {}/{}'.format(epoch + 1, n_epochs))

            for batch in range(n_train_batches):
                iteration += 1
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_observations)

                if self.verbose:
                    print('Batch {}/{}, size {}'.format(batch + 1, n_train_batches, end_idx - start_idx))

                # current_batch_size = end_idx - start_idx
                batch_idxs = train_idxs[start_idx:end_idx]
                train_loss, train_preds = self._process_batch(X_train[batch_idxs],
                                                              y_train[batch_idxs], iteration)
                train_metric_score = None
                if metric is not None:
                    train_metric_score = metric(train_preds, y_train[batch_idxs])
                    train_log['train_metric'].append(train_metric_score)

                train_log['train_loss'].append(train_loss)

                if self.verbose:
                    self._report(train_loss, train_metric_score, 'Train')

            if not (X_val is None or y_val is None):

                val_loss, val_preds, _ = self._forward(X_val, y_val, compute_loss_grad=False)
                val_metric_score = None

                if metric:
                    val_metric_score = metric(val_preds, y_val)
                    train_log['val_metric'].append(val_metric_score)
                train_log['val_loss'].append(val_loss)

                if self.verbose:
                    self._report(val_loss, val_metric_score, 'Val')

            for callback in on_epoch_callbacks:
                callback(model=self, log=train_log)

        return train_log

    def _report(self, loss, metric, type_error='Train'):
        """

        Print loss and metric scores.

        :param loss: loss score
        :param metric: metric score
        :param type_error: str, e.g. 'Train', or 'Val'
        """
        print('{} loss {}'.format(type_error, loss))
        if metric is not None:
            print('{} metric {}'.format(type_error, metric))
        print('--------------------------------------')

    def predict(self, X):
        """

        Output predictions of the model.

        :param X: input observations.
        :return: predictions of shape (n_observations, d1, d2, ...)
        """
        return self._forward(X, state='prediction')[1]

    def _process_batch(self, X, y, iteration=None):
        """

        Make forward and backward pass for the given input.

        :param X: input observations.
        :param y: labels
        :param iteration: number of current iteration, is mandatory for some optimizers, e.g. AdamOptimizer
        :return: 2-tuple of loss score and predictions for the given input.
        """
        train_loss, train_preds, loss_derivative = self._forward(X, y)
        self._backward(loss_derivative, iteration=iteration, return_grads=False)
        return train_loss, train_preds

    def _forward(self, X, y=None, compute_loss_grad=True, state='training'):
        """

        Make a forward pass.
        Note that if y is None, then loss score and loss grads will
            be None in output tuple.

        :param X: numpy.array, input data
        :param y: numpy.array, input labels. Default None. If None loss and loss gradient won`t be computed.
        :param compute_loss_grad: bool, indicates whether to compute loss derivatives
        :return: tuple, (loss_score, predictions, loss_grads).
            If y is None only predictions are not None.
        """
        for layer in self.layers:
            X = layer.forward(layer_input=X, state=state, seed=self.seed)

        if y is None:
            if self.loss and self.loss.name == "CrossEntropy":
                return softmax(X)
            return None, X, None

        loss, preds, dX = self.loss.build(X, y, compute_derivative=compute_loss_grad)

        return loss, preds, dX

    def _backward(self, dX, iteration=None, return_grads=False):
        """

        Make a backward pass (updates layer weights).
        Note that weight gradients can be returned,
        use this option carefully for high memory consumption.

        :param dX: input loss gradient
        :param iteration: number of current iteration, is mandatory for certain optimizers, e.g. Adam
        :param return_grads: if True, list of weight gradients for each layer is returned.
        :return: None if return_grads is False, otherwise see return_grads param.
        """
        grads = None
        if return_grads:
            grads = [None] * len(self.layers)

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            dW, dX = layer.backward(dX)
            if dW is not None:
                layer.weights = self.optimizer.update(layer.weights, dW,
                                                      self._get_weight_indices(i), iteration=iteration)
            if return_grads:
                grads[i] = dW

        return grads


def save_callback(frequency=1, path='', model_name='model'):
    """

    On epoch callback, save model to disk with the given frequency.

    :param frequency: indicates period (number of epochs),
        once upon which the model is saved.
    :param path: path to file.
    :param model_name: model name
    :return: on_epoch_callback with the interface f(model, log)
    """
    def save(model, log):
        current_epoch = log['current_epoch']
        current_metric = log['train_loss'][-1]
        if current_epoch % frequency == 0:
            filename = ''.join([model_name, '_', str(current_epoch), '_', str(current_metric)])
            filename = os.path.join(path, filename)
            with open(filename, 'wb') as file:
                pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)

    return save


# supports both, sparse and one-hot representations, but they should be passed row-wise - separate row corresponds for a
# separate observation
def accuracy_metric(preds, y):
    return np.sum(np.all(preds == y, axis=1)) / float(preds.shape[0])


def reduce_lr_plateau_callback(wait_epochs=10, monitor_metric='train_loss', reduce_factor=0.1):
    def reduce_lr_plateau(model, log):
        pass
