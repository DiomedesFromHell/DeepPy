import numpy as np
from initializers import UniformInitializer
from math_utils import softmax, sigmoid


# We always assume that final loss is a mean of the individual losses -
# sum of ind. losses for each observation in a batch divided by a batch size


# Layer Interface
class Layer(object):

    def __init__(self, input_shape=None, name=''):
        self.name = name
        self.input_shape = input_shape

    def initialize_parameters(self):
        raise NotImplementedError

    def forward(self, layer_input, *args, **kwargs):
        raise NotImplementedError

    def backward(self, layer_error, *args, **kwargs):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def size(self):
        return 0


class Affine(Layer):
    layer_type = 'Affine'

    def __init__(self, output_size, input_shape=(None, None),
                 weight_initializer=UniformInitializer(), bias_initializer=None, name='', reg=0.0):

        super(Affine, self).__init__(input_shape, name)
        # self.input_shape = input_shape  # first is batch size, second is dimensionality

        self.output_size = output_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weights = None  # matrix of concatenated column-wise weights (including biases)
        self.current_layer_input = None
        self.reg = reg

    @property
    def W(self):  # column wise weights (without biases)
        return None if self.weights is None else self.weights[:-1, :]

    @property
    def b(self):  # biases row
        return None if self.weights is None else self.weights[None, -1, :]

    def initialize_parameters(self):
        self.weight_initializer.shape = self.input_shape[1], self.output_size
        W_ = self.weight_initializer.initialize()
        if not self.bias_initializer:
            b_ = np.zeros((1, self.output_size))
        else:
            self.bias_initializer.shape = 1, self.output_size
            b_ = self.bias_initializer.initialize()
        self.weights = np.concatenate([W_, b_], axis=0)

    # output is batch_size x output_size
    def forward(self, layer_input, *args, **kwargs):
        self.current_layer_input = layer_input.copy()
        return layer_input @ self.W + self.b

    # layer_error is batch_size x output_size
    def backward(self, layer_error, *args, **kwargs):
        if layer_error.shape != (self.current_layer_input.shape[0], self.output_size):
            raise ValueError('Layer_error shape ({}, {}) does not match layer output shape ({}, {})'
                             .format(layer_error.shape[0], layer_error.shape[1], self.current_layer_input.shape[0],
                                     self.output_size))
        db = np.sum(layer_error, axis=0, keepdims=True)  # /n_observations
        dW = self.current_layer_input.T @ layer_error  # /n_observations
        dX = layer_error @ self.W.T
        dweights = np.concatenate([dW, db], axis=0)
        if self.reg > 0.:
            dweights += self.reg * self.weights
        return dweights, dX

    @property
    def output_shape(self):
        return self.input_shape[0], self.output_size

    @property
    def size(self):
        if self.weights is None:
            raise AttributeError("AffineLayer: weights should be initialized before calling size")
        return self.weights.size


class ConvolutionNaive(Layer):
    layer_type = 'Convolution'

    def __init__(self, n_filters, filter_size, strides=1,
                 input_shape=(None, None, None, None),
                 weight_initializer=UniformInitializer(),
                 padding=None, name=''):
        super(ConvolutionNaive, self).__init__(input_shape, name)
        self.ch = n_filters

        if isinstance(filter_size, int):
            self.filter_size = filter_size, filter_size
        else:
            self.filter_size = filter_size
        if isinstance(strides, int):
            self.strides = strides, strides
        else:
            self.strides = strides
        self.weight_initializer = weight_initializer
        self.padding = padding
        self.weights = None
        self.current_layer_input = None

    @property
    def ch_prev(self):
        return self.input_shape[3]

    def initialize_parameters(self):

        W_shape = self.filter_size[0], self.filter_size[1], self.ch_prev, self.ch

        self.weight_initializer.shape = W_shape
        self.weights = self.weight_initializer.initialize()

    def forward(self, layer_input, *args, **kwargs):
        output = np.zeros(self.output_shape)
        i, j = 0, 0

        for row in range(0, layer_input.shape[1] - self.filter_size[0] + 1, self.strides[0]):
            for col in range(0, layer_input.shape[2] - self.filter_size[1] + 1, self.strides[1]):
                X = layer_input[:, row:row + self.filter_size[0], col:col + self.filter_size[1], :]
                output[:, i, j, :] = np.tensordot(X, self.weights, ([1, 2, 3], [0, 1, 2]))

        self.current_layer_input = layer_input

        return output

    def backward(self, layer_error, *args, **kwargs):
        dW = np.zeros_like(self.weights)

        for i in range(layer_error.shape[1]):
            for j in range(layer_error.shape[2]):
                r1, c1 = i * self.strides[0], j * self.strides[1]
                r2, c2 = r1 + self.filter_size[0], c1 + self.filter_size[1]
                dW += np.einsum('bc, bkld -> kldc', layer_error[:, i, j, :],
                                self.current_layer_input[:, r1:r2, c1:c2, :])

        dX = np.zeros_like(self.current_layer_input)
        for i in range(layer_error.shape[1]):
            for j in range(layer_error.shape[2]):
                r1, c1 = i * self.strides[0], j * self.strides[1]
                r2, c2 = r1 + self.filter_size[0], c1 + self.filter_size[1]
                dX[:, r1:r2, c1:c2, :] += np.einsum('bc, kldc->bkld', layer_error[:, i, j, :], self.weights)

        return dW, dX

    @property
    def output_shape(self):
        return self.input_shape[0], 1 + (self.input_shape[1] - self.filter_size[0]) // self.strides[0], \
               1 + (self.input_shape[2] - self.filter_size[1]) // self.strides[1], self.ch

    @property
    def size(self):
        if self.weights is None:
            raise AttributeError("{}: weights should be initialized before calling size".format(self.__class__.__name__))
        return self.weights.size


class Flatten(Layer):

    layer_type = 'Flatten'

    #input shape is (batch_size, d1, d2, ...)
    def __init__(self, input_shape=None, name=''):
        super(Flatten, self).__init__(input_shape, name)

    def initialize_parameters(self):
        pass

    def forward(self, layer_input, *args, **kwargs):
        return layer_input.reshape(layer_input.shape[0], -1)

    def backward(self, layer_error, *args, **kwargs):
        shape = [layer_error.shape[0]]
        for i in range(1, len(self.input_shape)):
            shape.append(self.input_shape[i])
        return None, layer_error.reshape(shape)

    @property
    def output_shape(self):
        size = 1
        for i in range(1, len(self.input_shape)):
            size *= self.input_shape[i]
        return self.input_shape[0], size


class ReLuActivation(Layer):
    layer_type = 'ReLuActivation'

    def __init__(self, input_shape=None, name=''):
        super(ReLuActivation, self).__init__(input_shape, name)
        self.current_layer_input = None

    def forward(self, layer_input, *args, **kwargs):
        self.current_layer_input = layer_input.copy()
        output = layer_input
        output[layer_input < 0] = 0
        return output

    def backward(self, layer_error, *args, **kwargs):
        dH = np.copy(layer_error)
        dH[self.current_layer_input < 0] = 0
        return None, dH

    @property
    def output_shape(self):
        return self.input_shape

    def initialize_parameters(self, *args, **kwargs):
        pass


class SigmoidActivation(Layer):
    layer_type = 'SigmoidActivation'

    def __init__(self, input_shape=None, name=''):
        super(SigmoidActivation, self).__init__(input_shape, name)
        self.current_layer_input = None
        self.current_layer_output = None

    def forward(self, layer_input, *args, **kwargs):
        self.current_layer_input = layer_input
        self.current_layer_output = sigmoid(layer_input)
        return self.current_layer_output

    def backward(self, layer_error, *args, **kwargs):
        return None, layer_error * self.current_layer_output * (1 - self.current_layer_output)

    @property
    def output_shape(self):
        return self.input_shape

    def initialize_parameters(self):
        pass


class SoftmaxActivation(Layer):
    layer_type = 'SoftmaxActivation'

    def __init__(self, input_shape, name=''):
        super(SoftmaxActivation, self).__init__(input_shape, name)
        self.current_layer_input = None
        self.current_layer_output = None

    def forward(self, layer_input, *args, **kwargs):
        self.current_layer_input = layer_input
        self.current_layer_output = softmax(layer_input)
        return self.current_layer_output

    def backward(self, layer_error, *args, **kwargs):
        return None, layer_error * self.current_layer_output * (1 - self.current_layer_output)

    @property
    def output_shape(self):
        return self.input_shape

    def initialize_parameters(self):
        pass


if __name__ == '__main__':
    from utils.gradient_check import evaluate_gradient
    from math_utils import relative_error
    from losses import CrossEntropyLoss

    # n_features = 2
    batch_size = 7
    c = 3
    in_shape = (batch_size, c)
    X = (np.random.random_sample(in_shape[0] * in_shape[1]) * 10 - 5).reshape(in_shape)
    y = np.random.randint(c, size=batch_size)

    loss = CrossEntropyLoss()
    relu_layer = ReLuActivation()
    X_inp = relu_layer.forward(X)
    _, _, d = loss.build(X_inp, y)

    grad = relu_layer.backward(d)[1]

    f = lambda x: loss.build(x, y, compute_derivative=False)[0]
    num_grad = evaluate_gradient(f, X_inp)

    err, _ = relative_error(grad, num_grad)
    assert err < 1e6

    n_features = 10
    batch_size = 5
    c = 5
    in_shape = (batch_size, n_features)
    X = (np.random.random_sample(in_shape[0] * in_shape[1]) * 10 - 5).reshape(in_shape)
    y = np.random.randint(c, size=batch_size)

    n_neurons = c
    aff_layer = Affine(n_neurons)
    W = (np.random.random_sample((n_features + 1) * n_neurons) * 10 - 5).reshape(n_features + 1, n_neurons)
    aff_layer.weights = W

    X_inp = aff_layer.forward(X)
    _, _, d = loss.build(X_inp, y)
    grad_W, grad_X = aff_layer.backward(d)

    fW = lambda w: loss.build(aff_layer.forward(X), y)[0]
    num_grad_W = evaluate_gradient(fW, aff_layer.weights)

    # print(grad_W)
    # print('------')
    # print(num_grad_W)
    err, idx = relative_error(grad_W, num_grad_W)
    print(err, grad_W[idx], num_grad_W[idx])

    fX = lambda x: loss.build(aff_layer.forward(X), y)[0]
    num_grad_X = evaluate_gradient(fX, X)

    # print(grad_X)
    # print('------')
    # print(num_grad_X)
    err, idx = relative_error(grad_X, num_grad_X)
    print(err, grad_X[idx], num_grad_X[idx])

import tensorflow as tf

tf.keras.layers.Conv2D
