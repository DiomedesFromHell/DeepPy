import numpy as np
from initializers import UniformInitializer
from math_utils import softmax, sigmoid
from utils.array_view import window_view
import time

EPS = 0.0000001


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

        if not self.weight_initializer:
            W_ = np.zeros((self.input_shape[1], self.output_size))
        else:
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


class ConvolutionBase(Layer):
    layer_type = 'Convolution'

    def __init__(self, n_filters, filter_size, strides=1,
                 input_shape=(None, None, None, None),
                 use_biases=True,
                 weight_initializer=UniformInitializer(),
                 bias_initializer=None,
                 padding=None, name=''):
        super(ConvolutionBase, self).__init__(input_shape, name)
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
        self.bias_initializer = bias_initializer
        self.padding = padding if padding == 'same' else None
        self.W_shape = None
        self.b_shape = None
        self.use_biases = use_biases
        self.weights = None
        self.current_layer_input = None

    @property
    def ch_prev(self):
        return self.input_shape[3]

    @property
    def W(self):
        return self.weights[:-1, :]

    @property
    def b(self):
        return self.weights[None, -1, :]

    # pad input to preserve width and height of image
    def pad_input_same(self, X):
        pad_width = [(0, 0), ]
        for i, stride in enumerate(self.strides):
            delta = (self.input_shape[i + 1] - 1) * stride + self.filter_size[i] - self.input_shape[i + 1]
            d = delta // 2 if delta % 2 == 0 else (delta + 1) // 2
            pad_width.append((d, d))
        pad_width.append((0, 0))
        print(pad_width)
        X_padded = np.pad(X, pad_width=pad_width)
        return X_padded

    def initialize_parameters(self):

        self.weights = np.zeros((np.prod(self.filter_size) * self.ch_prev + 1, self.ch))

        if self.weight_initializer:
            self.weight_initializer.shape = self.weights.shape[0] - 1, self.weights.shape[1]
            self.weights[:-1, :] = self.weight_initializer.initialize()

        if self.use_biases and self.bias_initializer:
            self.weights[-1, :] = self.bias_initializer.initialize()

    def forward(self, layer_input, *args, **kwargs):
        t0 = time.time()
        batch_size = layer_input.shape[0]

        if self.padding == 'same':
            layer_input = self.pad_input_same(layer_input)

        print('Padding takes: ', time.time() - t0)
        output = np.zeros(self._output_shape_for_batch(batch_size))
        i = 0

        for row in range(0, layer_input.shape[1] - self.filter_size[0] + 1, self.strides[0]):
            j = 0
            for col in range(0, layer_input.shape[2] - self.filter_size[1] + 1, self.strides[1]):
                X = layer_input[:, row:row + self.filter_size[0], col:col + self.filter_size[1], :].reshape(batch_size,
                                                                                                            -1)
                output[:, i, j, :] = X @ self.W
                if self.use_biases:
                    output[:, i, j, :] += self.b
                j += 1
            i += 1

        self.current_layer_input = layer_input
        print('Naive forward takes: ', time.time() - t0)
        return output

    def backward(self, layer_error, *args, **kwargs):
        t = time.time()
        dW = np.zeros_like(self.weights)

        if self.use_biases:
            dW[-1, :] = np.sum(layer_error, axis=(0, 1, 2))

        for i in range(layer_error.shape[1]):
            for j in range(layer_error.shape[2]):
                r1, c1 = i * self.strides[0], j * self.strides[1]
                r2, c2 = r1 + self.filter_size[0], c1 + self.filter_size[1]
                dW[:-1, :] += np.einsum('bc, bkld -> kldc', layer_error[:, i, j, :],
                                        self.current_layer_input[:, r1:r2, c1:c2, :]).reshape(-1, self.ch)

        dX = np.zeros_like(self.current_layer_input)
        patch_shape = self.filter_size[0], self.filter_size[1], self.ch_prev, self.ch
        for i in range(layer_error.shape[1]):
            for j in range(layer_error.shape[2]):
                r1, c1 = i * self.strides[0], j * self.strides[1]
                r2, c2 = r1 + self.filter_size[0], c1 + self.filter_size[1]

                dX[:, r1:r2, c1:c2, :] += np.einsum('bc, kldc->bkld', layer_error[:, i, j, :],
                                                    self.weights[:-1, :].reshape(patch_shape))

        print('Naive backward takes: ', time.time() - t)
        return dW if self.use_biases else dW[:-1, :], dX[:, 1:-1, 1:-1, :] if self.padding == 'same' else dX

    @property
    def output_shape(self):
        return self._output_shape_for_batch(self.input_shape[0])

    def _output_shape_for_batch(self, batch_size):

        if self.padding == 'same':
            return batch_size, self.input_shape[1], self.input_shape[2], self.ch

        return batch_size, 1 + (self.input_shape[1] - self.filter_size[0]) // self.strides[0], \
               1 + (self.input_shape[2] - self.filter_size[1]) // self.strides[1], self.ch

    @property
    def size(self):
        if self.weights is None:
            raise AttributeError(
                "{}: weights should be initialized before calling size".format(self.__class__.__name__))
        return self.weights.size if self.use_biases else self.weights.size - self.weights.shape[1]


class Convolution(ConvolutionBase):

    def _convolve(self, X, F, strides):
        """

        2d convolution, 3d window slides over 1st and 2nd dims of X (channels are summed up)

        :param X: tensor to be convolved, expected shape is (batch, H, W, channels)
        :param F: filter, expected shape is (n_filters, channels, h, w)
        :param strides: sliding step in each direction
        :return: matrix, result of convolution
        """
        X_w = window_view(X, strides, (F.shape[2], F.shape[3]), axes=(1, 2), as_contiguous=True)
        output = np.tensordot(X_w, F, axes=((4, 5, 3), (2, 3, 0)))
        return output

    def forward(self, layer_input, *args, **kwargs):

        t0 = time.time()
        batch_size = layer_input.shape[0]
        if self.padding == 'same':
            layer_input = self.pad_input_same(layer_input)

        print('Padding takes: ', time.time() - t0)

        t = time.time()
        X_w = window_view(layer_input, self.strides, self.filter_size, axes=(1, 2))
        self.current_layer_input = X_w
        # print('Window view takes: ', time.time() - t)
        # t = time.time()
        # X_w = X_w.reshape(
        #     (batch_size, X_w.shape[1] * X_w.shape[2], -1))
        # print('Reshape input takes: ', time.time() - t)
        # t = time.time()
        # W_sh = self.W.reshape((self.ch, self.ch_prev, self.filter_size[0], self.filter_size[1]))
        # print('Filter reshape takes: ', time.time() - t)
        t = time.time()
        output = np.tensordot(X_w, self.W.reshape((self.ch, self.ch_prev, self.filter_size[0], self.filter_size[1])),
                              axes=((4, 5, 3), (2, 3, 1)))
        print('Tensor product takes: ', time.time() - t)
        # output = np.einsum('bwhckf, dckf->bwhd', X_w, self.W.reshape((self.ch, self.ch_prev, self.filter_size[0], self.filter_size[1])))
        # t = time.time()
        # output = X_w.dot(self.W)
        # print('Matrix product takes: ', time.time() - t)
        if self.use_biases:
            output += self.b
        t = time.time()
        output = output.reshape(self._output_shape_for_batch(batch_size))
        print('Output reshape takes: ', time.time() - t)
        print('Fast forward takes: ', time.time() - t0)
        return output

    def backward(self, layer_error, *args, **kwargs):
        t0 = time.time()
        dW = np.zeros_like(self.weights)

        if self.use_biases:
            dW[-1, :] = np.sum(layer_error, axis=(0, 1, 2))

        batch_size, h_out, w_out, ch = layer_error.shape
        t = time.time()
        dW[:-1, :] = np.tensordot(layer_error, self.current_layer_input, axes=((0, 1, 2), (0, 1, 2))).reshape(-1,
                                                                                                              self.ch)
        # dW[:-1, :] = np.einsum('bwhd, bwhcij->ijcd', layer_error, self.current_layer_input).reshape(-1, self.ch)
        print('Back dw einsum takes: ', time.time() - t)

        W_rot180 = np.rot90(self.W.reshape((self.ch, self.ch_prev, self.filter_size[0], self.filter_size[1])),
                            axes=(2, 3))
        t = time.time()
        sparse_error = np.zeros((batch_size, h_out * self.strides[0], w_out * self.strides[1], ch))
        # sparse_error.as_st
        # idx = np.meshgrid()
        idx = np.ix_(np.arange(batch_size), self.strides[0] * np.arange(h_out), self.strides[1] * np.arange(w_out),
                     np.arange(ch))
        sparse_error[idx] = layer_error
        del layer_error
        print('build sparse takes: ', time.time() - t)
        pad_width = (
            (0, 0), (self.filter_size[0] - 1, self.filter_size[0] - 1),
            (self.filter_size[1] - 1, self.filter_size[1] - 1),
            (0, 0))
        sparse_error = np.pad(sparse_error, pad_width)
        t = time.time()
        dX = self._convolve(sparse_error, W_rot180, (1, 1))
        print('back conv takes: ', time.time() - t)
        print('Fast backward takes: ', time.time() - t0)
        return dW if self.use_biases else dW[:-1, :], dX[:, 1:-1, 1:-1, :] if self.padding == 'same' else dX


class MaxPool(Layer):
    layer_type = 'MaxPool'

    def __init__(self, filter_size, input_shape=(None, None, None, None), name=''):
        self.filter_size = filter_size
        self.input_shape = input_shape
        super(MaxPool, self).__init__(input_shape, name)
        self.pad_w = 0
        self.pad_h = 0
        self.current_activation_map = None

    def initialize_parameters(self):
        rh, rw = self.input_shape[1] % self.filter_size[0], self.input_shape[2] % self.filter_size[1]
        if rh != 0:
            add_h = self.filter_size[0] - rh
            add_h = add_h if add_h % 2 == 0 else add_h + 1
            self.pad_h = add_h // 2
        if rh != 0:
            add_w = self.filter_size[1] - rw
            add_w = add_w if add_w % 2 == 0 else add_w + 1
            self.pad_w = add_w // 2

    def forward(self, layer_input, *args, **kwargs):
        pad_width = ((0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w), (0, 0))
        layer_input = np.pad(layer_input, pad_width)

        X_w = window_view(layer_input, steps=self.filter_size, window_size=self.filter_size, axes=(1, 2),
                          as_contiguous=True)
        output = np.amax(X_w, axis=(4, 5), keepdims=True)
        self.current_activation_map = X_w != output

        return output

    def backward(self, layer_error, *args, **kwargs):
        print('Maxpool back starts')
        dX = np.repeat(layer_error, self.filter_size[0], axis=1)
        dX = np.repeat(dX, self.filter_size[1], axis=2)
        indices = np.where(self.current_activation_map)
        print('Mask to row indices finished')
        # print(indices.shape)
        indices = list(indices)
        indices[1] *= indices[-2]
        indices[2] *= indices[-1]

        print('Multiplying finished')
        dX[indices[:-2]] = 0.
        return None, dX[:, self.pad_h:dX.shape[1] - self.pad_h, self.pad_w:dX.shape[2] - self.pad_w, :]

    @property
    def output_shape(self):

        h, rh = divmod(self.input_shape[1], self.filter_size[0])
        w, rw = divmod(self.input_shape[2], self.filter_size[1])
        if rh != 0:
            h += 1
        if rw != 0:
            w += 1
        return self.input_shape[0], h, w, self.input_shape[3]


class Flatten(Layer):
    layer_type = 'Flatten'

    # input shape is (batch_size, d1, d2, ...)
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


class BatchNormalization(Layer):
    layer_type = 'BatchNormalization'

    def __init__(self, input_shape=None, name='', alpha=0.9):
        super(BatchNormalization, self).__init__(input_shape, name)
        self.weights = None
        self.current_std = None
        self.current_input_centr = None
        self.variance = None
        self.mean_tot = None
        self.variance_tot = None
        self.alpha = alpha

    @property
    def a(self):
        return self.weights[0, :]

    @property
    def b(self):
        return self.weights[1, :]

    def initialize_parameters(self):
        shape = [2]
        shape += [self.input_shape[i] for i in range(1, len(self.input_shape))]
        self.weights = np.ones(shape)

    def forward(self, layer_input, *args, **kwargs):
        state = kwargs.get('state', 'prediction')

        batch_size = float(layer_input.shape[0])
        if state == 'training':
            mean_batch = np.sum(layer_input, axis=0, keepdims=True) / batch_size
            self.current_input_centr = layer_input - mean_batch
            var_batch = np.sum(self.current_input_centr ** 2, axis=0, keepdims=True) / batch_size
            if self.mean_tot is None:
                self.mean_tot = mean_batch
            else:
                self.mean_tot = self.alpha * self.mean_tot + (1 - self.alpha) * mean_batch
            if self.variance_tot is None:
                self.variance_tot = var_batch
            else:
                self.variance_tot = self.alpha * self.variance_tot + (1 - self.alpha) * var_batch

            self.current_std = (var_batch + EPS) ** 0.5
            return self.a * self.current_input_centr / self.current_std + self.b
        if self.mean_tot is None or self.variance_tot is None:
            mean_batch = np.sum(layer_input, axis=0, keepdims=True) / batch_size
            curr_inp_centr = layer_input - mean_batch
            var_batch = np.sum(curr_inp_centr ** 2, axis=0, keepdims=True) / batch_size
            return self.a * curr_inp_centr / (var_batch + EPS) ** 0.5 + self.b

        return self.a * (layer_input - self.mean_tot) / (self.variance_tot + EPS) ** 0.5 + self.b

    def backward(self, layer_error, *args, **kwargs):
        batch_size = layer_error.shape[0]
        dx_tr = layer_error * self.a
        dvar = np.sum(-0.5 * dx_tr * self.current_input_centr / self.current_std ** 3, axis=0, keepdims=True)
        dmean = np.sum(-dx_tr / self.current_std, axis=0, keepdims=True) - 2 * dvar * np.sum(
            self.current_input_centr, axis=0, keepdims=True) / batch_size
        dx = dx_tr / self.current_std + (2 * dvar * self.current_input_centr + dmean) / batch_size
        dweights = np.stack(
            [np.sum(layer_error * self.current_input_centr / self.current_std, axis=0), np.sum(layer_error, axis=0)])
        return dweights, dx

    @property
    def output_shape(self):
        return self.input_shape

    @property
    def size(self):
        if self.weights is None:
            raise AttributeError(
                "{}: weights should be initialized before calling size".format(self.__class__.__name__))
        return self.weights.size


class DropoutLayer(Layer):
    layer_type = 'Droupout'

    def __init__(self, input_shape=None, name='', drop_rate=0.3):
        super(DropoutLayer, self).__init__(input_shape, name)
        self.drop_rate = drop_rate
        self.U = None

    def forward(self, layer_input, *args, **kwargs):
        state = kwargs.get('state', 'prediction')
        if isinstance(kwargs['seed'], int):
            np.random.seed(kwargs['seed'])
        # self.U = np.ones_like(layer_input)
        self.U = (1./1-self.drop_rate)*(np.random.rand(*layer_input.shape) >= self.drop_rate)
        if state == 'prediction':
            return layer_input
        else:
            return self.U*layer_input

    def backward(self, layer_error, *args, **kwargs):
        return None, layer_error*self.U

    @property
    def output_shape(self):
        return self.input_shape

    def initialize_parameters(self):
        if len(self.input_shape) != 2:
            raise AttributeError(
                'Droupout should be used only for dense layers. Called with input_shape={}'.format(self.input_shape))


def pad_input_same(X, filter_size, strides):
    """

    :param X: input data of shape (batch_size, height, width, n_channels)
    :param filter_size: filter size of shape (filter_height, filter_width)
    :param strides: step size for each of strided dimensions, which are height and width
    :return: initial array padded with zeros, in a way that after sliding with a window of filter_size with
        the given strides output shape will be the same as X.shape
    """
    input_shape = X.shape
    pad_width = [(0, 0), ]
    for i, stride in enumerate(strides):
        delta = (input_shape[i + 1] - 1) * stride + filter_size[i] - input_shape[i + 1]
        d = delta // 2 if delta % 2 == 0 else (delta + 1) // 2
        pad_width.append((d, d))
    pad_width.append((0, 0))
    print(pad_width)
    X_padded = np.pad(X, pad_width=pad_width)
    return X_padded


def check_batch_normalization():
    from utils.gradient_check import evaluate_gradient
    from math_utils import relative_error
    # means = [0.3, -4, 17]
    # stds = [0.15, 2, 30]
    # X = []
    # d = 10000
    # for i in range(3):
    #     X.append(np.random.normal(means[i], stds[i], d))
    # X = np.array(X).T
    #
    # means = np.array(means).reshape((1, 3))
    # stds = np.array(stds).reshape((1, 3))
    #
    # layer = BatchNormalization(input_shape=(50, 3))
    # layer.initialize_parameters()
    # xs = X - np.mean(X, axis=0, keepdims=True)
    # v = np.sum(xs ** 2, axis=0, keepdims=True) / d
    # layer.weights[1, :] = np.zeros(3)
    # out = layer.forward(X, mode='prediction')
    #
    # # print((X-means)/stds)
    # # print(out)
    # np.random.seed(231)
    # N, D = 4, 5
    # x = 5 * np.random.randn(N, D) + 12
    # gamma = np.random.randn(D)
    # beta = np.random.randn(D)
    # w = np.vstack([beta, gamma])
    # layer.weights = w
    # print(w.shape)
    #
    # dout = np.random.randn(N, D)
    #
    # bn_param = {'mode': 'train'}
    # fx = lambda x: layer.forward(x, state='training')
    # # fg = lambda a: batchnorm_forward(x, a, beta, bn_param)[0]
    # # fb = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]
    #
    # dx_num = evaluate_gradient(fx, x, dout)
    # # da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
    # # db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)
    # # print(np.isclose(x-np.mean(x, axis=0, keepdims=True), layer.current_input_centr))
    # dw, dx = layer.backward(dout)
    # # You should expect to see relative errors between 1e-13 and 1e-8
    # print(dx)
    # print(dx_num)
    # print('dx error: ', relative_error(dx_num, dx))
    # # print('dgamma error: ', relative_error(da_num, dgamma))
    # # print('dbeta error: ', relative_error(db_num, dbeta))
    # np.random.seed(231)
    # x = np.random.randn(500, 500) + 10
    #
    # for p in [0.25, 0.4, 0.7]:
    #     dropout_layer = DropoutLayer(drop_rate=1-p)
    #     out = dropout_layer.forward(x, state='training')
    #     out_test = dropout_layer.forward(x, state='prediction')
    #
    #     print('Running tests with p = ', p)
    #     print('Mean of input: ', x.mean())
    #     print('Mean of train-time output: ', out.mean())
    #     print('Mean of test-time output: ', out_test.mean())
    #     print('Fraction of train-time output set to zero: ', (out == 0).mean())
    #     print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
    #     print()

    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)
    dropout_layer = DropoutLayer(drop_rate=0.8)
    # dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
    out = dropout_layer.forward(x, state='training', seed=123)
    _, dx = dropout_layer.backward(dout)
    dx_num = evaluate_gradient(lambda xx: dropout_layer.forward(xx, state='training', seed=123), x, dout)

    print(dx)
    print(dx_num)
    # Error should be around e-10 or less
    print('dx relative error: ', relative_error(dx, dx_num))


if __name__ == '__main__':
    check_batch_normalization()
