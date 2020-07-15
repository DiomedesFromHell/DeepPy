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
        batch_size = layer_input.shape[0]

        if self.padding == 'same':
            layer_input = self.pad_input_same(layer_input)

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

        return output

    def backward(self, layer_error, *args, **kwargs):
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

    def forward(self, layer_input, *args, **kwargs):
        pass

    def backward(self, layer_error, *args, **kwargs):
        pass


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


if __name__ == '__main__':
    from utils.gradient_check import evaluate_gradient
    from math_utils import relative_error
    from losses import CrossEntropyLoss

    # # n_features = 2
    # batch_size = 7
    # c = 3
    # in_shape = (batch_size, c)
    # X = (np.random.random_sample(in_shape[0] * in_shape[1]) * 10 - 5).reshape(in_shape)
    # y = np.random.randint(c, size=batch_size)
    #
    # loss = CrossEntropyLoss()
    # relu_layer = ReLuActivation()
    # X_inp = relu_layer.forward(X)
    # _, _, d = loss.build(X_inp, y)
    #
    # grad = relu_layer.backward(d)[1]
    #
    # f = lambda x: loss.build(x, y, compute_derivative=False)[0]
    # num_grad = evaluate_gradient(f, X_inp)
    #
    # err, _ = relative_error(grad, num_grad)
    # assert err < 1e6
    #
    # n_features = 10
    # batch_size = 5
    # c = 5
    # in_shape = (batch_size, n_features)
    # X = (np.random.random_sample(in_shape[0] * in_shape[1]) * 10 - 5).reshape(in_shape)
    # y = np.random.randint(c, size=batch_size)
    #
    # n_neurons = c
    # aff_layer = Affine(n_neurons)
    # W = (np.random.random_sample((n_features + 1) * n_neurons) * 10 - 5).reshape(n_features + 1, n_neurons)
    # aff_layer.weights = W
    #
    # X_inp = aff_layer.forward(X)
    # _, _, d = loss.build(X_inp, y)
    # grad_W, grad_X = aff_layer.backward(d)
    #
    # fW = lambda w: loss.build(aff_layer.forward(X), y)[0]
    # num_grad_W = evaluate_gradient(fW, aff_layer.weights)
    #
    # # print(grad_W)
    # # print('------')
    # # print(num_grad_W)
    # err, idx = relative_error(grad_W, num_grad_W)
    # print(err, grad_W[idx], num_grad_W[idx])
    #
    # fX = lambda x: loss.build(aff_layer.forward(X), y)[0]
    # num_grad_X = evaluate_gradient(fX, X)
    #
    # # print(grad_X)
    # # print('------')
    # # print(num_grad_X)
    # err, idx = relative_error(grad_X, num_grad_X)
    # print(err, grad_X[idx], num_grad_X[idx])

    h, w = 4, 4
    f = 3
    c = 3
    b = 2
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    # x_shape = (b, h, w, c)
    # w_shape = (h, w, c, f)
    layer = ConvolutionNaive(f, (h, w), strides=2)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    x = np.transpose(x, [0, 2, 3, 1])
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    w = np.transpose(w, [2, 3, 1, 0])

    b = np.linspace(-0.1, 0.2, num=3).reshape(1, -1)
    w = w.reshape(-1, 3)
    layer.weights = np.concatenate([w, b], axis=0)
    conv_param = {'stride': 2, 'pad': 1}
    # out, _ = conv_forward_naive(x, w, b, conv_param)
    x_new = np.zeros((2, x.shape[1] + 2, x.shape[2] + 2, 3))
    layer.input_shape = x_new.shape
    x_new[:, 1:-1, 1:-1, :] = x
    out = layer.forward(x_new)
    print(out.shape)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216]],
                             [[0.21027089, 0.21661097],
                              [0.22847626, 0.23004637]],
                             [[0.50813986, 0.54309974],
                              [0.64082444, 0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[0.69108355, 0.66880383],
                              [0.59480972, 0.56776003]],
                             [[2.36270298, 2.36904306],
                              [2.38090835, 2.38247847]]]])

    # Compare your output to ours; difference should be around e-8
    print('Testing conv_forward_naive')
    print(out)
    # print('difference: ', rel_error(out, correct_out))
