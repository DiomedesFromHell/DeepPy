import numpy as np


class Initializer(object):

    def initialize(self):
        raise NotImplementedError


class UniformInitializer(Initializer):

    def __init__(self, shape=None, start=-1, end=1):
        self.start = start
        self.end = end
        self.shape = shape

    def initialize(self):
        size = np.prod(self.shape)
        return (np.random.random_sample(size) * (self.end - self.start) + self.start).reshape(self.shape)


class NormalInitializer(Initializer):

    def __init__(self, shape=None, std=0.001, mean=0.0):
        self.shape = shape
        self.std = std
        self.mean = mean

    def initialize(self):
        return np.random.normal(self.mean, self.std, self.shape)


class HeInitializer(Initializer):

    def __init__(self, shape=None, mode='forward'):
        self.mode = mode
        self.shape = shape

    def initialize(self):
        n_input_units, n_output_units = self._compute_units(self.shape)
        if self.mode == 'forward':
            std = (2. / n_input_units) ** 0.5
        elif self.mode == 'backward':
            std = (2. / n_output_units) ** 0.5
        elif self.mode == 'average':
            std = (2. / (n_input_units + n_output_units)) ** 0.5
        else:
            raise ValueError("HeInitializer: unsupported mode {}.".format(self.mode))
        return np.random.normal(0, std, self.shape)

    # inspired by tf.keras, we assume that for convolutional layers shape is (..., n_input_channels, n_output_channels)
    def _compute_units(self, shape):
        if len(shape) == 2:
            n_input_units = shape[0]
            n_output_units = shape[1]
            return n_input_units, n_output_units
        else:
            raise ValueError("HeInitializer: Unsupported layer weights shape {}".format(shape))
