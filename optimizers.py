import numpy as np

# small number for numerical stability
from math_utils import relative_error

C_NUM = 1e-8


class Optimizer(object):

    def __init__(self, learning_rate=0.001, size=None):
        self.learning_rate = learning_rate
        self.size = size
        self.name = self.__class__.__name__

    def update(self, weights, dweights, weights_indices_range, *args, **kwargs):
        raise NotImplementedError()

    def _validate_update_input(self, weights, dweights, weights_indices_range=None):

        if weights.size != dweights.size:
            raise ValueError("{}: weights size {} and dweights {} should be equal."
                             .format(self.name, weights.size, dweights.size))
        if weights_indices_range is not None:
            r = weights_indices_range[1] - weights_indices_range[0]
            if r != weights.size:
                raise ValueError("{}: weights indices range {} is not equal to weights size {}."
                                 .format(self.name, r, weights.size))
            if weights_indices_range[0] < 0 or weights_indices_range[1] > self.size:
                raise ValueError("{}: weight_indices_range {} is out of {}.size {}."
                                 .format(self.name, weights_indices_range, self.name, self.size))


class GradientDescent(Optimizer):

    def __init__(self, learning_rate=0.001, size=None):
        super(GradientDescent, self).__init__(learning_rate, size)

    def update(self, weights, dweights, weights_indices_range, *args, **kwargs):
        self._validate_update_input(weights, dweights)
        return weights - self.learning_rate * dweights


class SGDMomentum(Optimizer):

    def __init__(self, learning_rate=0.001, mu=0.9, size=None):
        super(SGDMomentum, self).__init__(learning_rate, size)
        self.mu = mu
        self.velocities = None

    def update(self, weights, dweights, weights_indices_range, *args, **kwargs):
        self._validate_update_input(weights, dweights, weights_indices_range)

        if self.velocities is None:
            self.velocities = np.zeros(self.size, dtype='float64')

        start_idx, end_idx = weights_indices_range

        self.velocities[start_idx:end_idx] *= self.mu
        self.velocities[start_idx:end_idx] -= self.learning_rate * dweights.flatten()
        weights += self.velocities[start_idx:end_idx].reshape(weights.shape)
        return weights


class SGDNesterovMomentum(SGDMomentum):

    def update(self, weights, dweights, weights_indices_range, *args, **kwargs):
        self._validate_update_input(weights, dweights, weights_indices_range)

        if self.velocities is None:
            self.velocities = np.zeros(self.size, dtype='float64')
        start_idx, end_idx = weights_indices_range

        v_old = np.copy(self.velocities[start_idx:end_idx])
        self.velocities[start_idx:end_idx] *= self.mu
        self.velocities[start_idx:end_idx] -= self.learning_rate * dweights.flatten()
        weights += ((1. + self.mu) * self.velocities[start_idx:end_idx] - self.mu * v_old).reshape(weights.shape)
        return weights


class RMSProp(Optimizer):

    def __init__(self, learning_rate=0.001, decay=0.9, size=None):
        super(RMSProp, self).__init__(learning_rate, size)
        self.decay = decay
        self.grad_squared = None

    def update(self, weights, dweights, weights_indices_range, *args, **kwargs):
        self._validate_update_input(weights, dweights, weights_indices_range)

        if self.grad_squared is None:
            self.grad_squared = np.zeros(self.size, dtype='float64')
        start_idx, end_idx = weights_indices_range[0], weights_indices_range[1]

        self.grad_squared[start_idx:end_idx] *= self.decay
        self.grad_squared[start_idx:end_idx] += (1. - self.decay) * (dweights ** 2).flatten()
        weights -= (self.learning_rate * dweights.flatten() / (C_NUM + self.grad_squared[start_idx:end_idx] ** 0.5)) \
            .reshape(weights.shape)
        return weights


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, decay_fst_mom=0.9, decay_sec_mom=0.999, size=None):
        super(Adam, self).__init__(learning_rate, size)
        self.alpha1 = decay_fst_mom
        self.alpha2 = decay_sec_mom
        self.first_momentum = None
        self.second_momentum = None
        self.t_prev = 0
        self.alpha1_pow_cache = 1
        self.alpha2_pow_cache = 1
        self.is_use_cache = True

    def update(self, weights, dweights, weights_indices_range, *args, **kwargs):
        if 'iteration' not in kwargs:
            raise ValueError('Adam: iteration was not passed.')
        t = kwargs['iteration']
        self._validate_update_input(weights, dweights, weights_indices_range)

        if self.first_momentum is None or self.second_momentum is None:
            self.first_momentum = np.zeros(self.size)
            self.second_momentum = np.zeros(self.size)
        start_idx, end_idx = weights_indices_range

        self.first_momentum[start_idx:end_idx] *= self.alpha1
        self.first_momentum[start_idx:end_idx] += (1. - self.alpha1) * dweights.flatten()

        self.second_momentum[start_idx:end_idx] *= self.alpha2
        self.second_momentum[start_idx:end_idx] += (1. - self.alpha2) * (dweights ** 2).flatten()

        if t == self.t_prev + 1:
            self.t_prev += 1
            self.alpha1_pow_cache *= self.alpha1
            self.alpha2_pow_cache *= self.alpha2
            fm_corrector = 1. - self.alpha1_pow_cache
            sm_corrector = 1. - self.alpha2_pow_cache
        else:
            fm_corrector = 1. - self.alpha1 ** t
            sm_corrector = 1. - self.alpha2 ** t
        fm_corrected = self.first_momentum[start_idx:end_idx] / fm_corrector
        sm_corrected = self.second_momentum[start_idx:end_idx] / sm_corrector

        weights = weights - (self.learning_rate * fm_corrected / (C_NUM + sm_corrected ** 0.5)) \
            .reshape(weights.shape)

        return weights


if __name__ == '__main__':
    # shape = 4, 4
    # size = shape[0] * shape[1]
    # optimizer = SGDMomentum(size=size)
    # grads = np.arange(4)
    # optimizer.update(np.zeros((2, 3)), np.arange(6), (11, 17))

    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
    v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)

    learning_rate = 1e-2
    optimizer = Adam(learning_rate=learning_rate, size=w.size)
    optimizer.t = 6 * np.ones(w.size)
    optimizer.first_momentum = m.flatten()
    optimizer.second_momentum = v.flatten()
    config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
    next_w = optimizer.update(w, dw, (0, w.size), iteration=6)
    next_m = optimizer.first_momentum.reshape(w.shape)
    next_v = optimizer.second_momentum.reshape(w.shape)
    expected_next_w = np.asarray([
        [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
        [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
        [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
        [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([
        [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853, ],
        [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385, ],
        [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767, ],
        [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966, ]])
    expected_m = np.asarray([
        [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
        [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
        [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
        [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])

    # You should see relative errors around e-7 or less
    err, idx = relative_error(expected_next_w, next_w)

    # print(expected_next_w)
    # print(next_w)
    print('next_w error: ', err, expected_next_w[idx], next_w[idx])
    # print('v error: ', rel_error(expected_v, config['v']))
    # print('m error: ', rel_error(expected_m, config['m']))
    dw1 = np.array([[-0.6, -0.54736842, -0.49473684, -0.44210526, -0.38947368],
           [-0.33684211, -0.28421053, -0.23157895, -0.17894737, -0.12631579],
           [-0.07368421, -0.02105263, 0.03157895, 0.08421053, 0.13684211],
           [0.18947368, 0.24210526, 0.29473684, 0.34736842, 0.4]])
    # w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    m_n = 0.9*m + 0.1*dw1
    v_n = 0.999*v + 0.001*dw1**2
    mc = m_n/(1 - 0.9**6.)
    print(mc)
    vc = v_n/(1 - 0.999**6.)
    print(vc)
    wn = w - 0.01*mc/(1e-8 + vc**0.5)
    print(wn)
    # [[1.17213255 1.21968617 1.2672398  1.31479342 1.36234704]
    #  [1.40990066 1.45745429 1.50500791 1.55256153 1.60011516]
    # [1.64766878
    # 1.6952224
    # 1.74277603
    # 1.79032965
    # 1.83788327]
    # [1.88543689 1.93299052 1.98054414 2.02809776 2.07565139]]
    #
    # [[140.212144   138.09267384 135.97431393 133.85706428 131.74092487]
    #  [129.62589572 127.51197681 125.39916816 123.28746976 121.17688161]
    # [119.06740372
    # 116.95903607
    # 114.85177868
    # 112.74563154
    # 110.64059465]
    # [108.53666801 106.43385162 104.33214548 102.2315496  100.13206396]]