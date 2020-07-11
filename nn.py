import numpy as np
from collections import namedtuple


class MLP(object):

    OUTPUT_ACTIVATIONS = ['softmax']
    HIDDEN_ACTIVATIONS = ['sigmoid']
    COST_FUNCTIONS = ['cross-entropy']

    def __init__(self, hidden_sizes, hidden_activation, output_activation, cost_function):
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hid_num = len(self.hidden_sizes)
        self.num_layers = self.hid_num + 2
        self.cost_function = cost_function


    def init_parameters_normal(self, inp_dim, out_dim):
        self.W = []
        self.b = []
        s = self.hidden_sizes[0]*inp_dim
        for i in range(1, self.hid_num):
            s += self.hidden_sizes[i]*self.hidden_sizes[i-1]
        s += out_dim*self.hidden_sizes[-1]
        print('total_number_of_weights: '+str(s))
        weights = np.random.normal(size=s)

        l = self.hidden_sizes[0]*inp_dim
        self.W.append(np.array(weights[0:l]).reshape((self.hidden_sizes[0], inp_dim)))
        self.b.append(np.zeros((self.hidden_sizes[0], 1)))

        for i in range(1, self.hid_num):
            r = self.hidden_sizes[i]*self.hidden_sizes[i-1]
            self.W.append(np.array(weights[l:r]).reshape(self.hidden_sizes[i], self.hidden_sizes[i-1]))
            self.b.append(np.zeros((self.hidden_sizes[i], 1)))
            l = r
            assert self.W[i].shape[0] == self.b[i].shape[0]
            assert self.W[i].shape[1] == self.W[i-1].shape[0]
        r = l + out_dim * self.hidden_sizes[-1]
        self.W.append(np.array(weights[l:r]).reshape((out_dim, self.hidden_sizes[-1])))
        self.b.append(np.zeros((out_dim, 1)))
        assert self.W[0].shape[0] == self.b[0].shape[0]
        assert self.W[-1].shape[0] == self.b[-1].shape[0]
        assert len(self.W) == len(self.b) == self.hid_num + 1


    def softmax(self, z):
        z_max = np.max(z, axis=0, keepdims=True)
        return np.exp(z-z_max)/(np.sum(np.exp(z-z_max), axis=0, keepdims=True))


    def sigmoid(self, z):
        pos_mask = (z >= 0)
        neg_mask = (z < 0)

        x = np.zeros_like(z, dtype=float)
        x[pos_mask] = np.exp(-z[pos_mask])
        x[neg_mask] = np.exp(z[neg_mask])

        top = np.ones_like(z, dtype=float)
        top[neg_mask] = x[neg_mask]
        res = top / (1 + x)
        return res


    def activation(self, z, activation_type='sigmoid'):
        if activation_type == 'sigmoid':
            res = self.sigmoid(z)
        elif activation_type == 'softmax':
            res = self.softmax(z)
        return res


    def activation_derivative(self, z, activation_type):
        if activation_type == 'sigmoid':
            g = self.sigmoid(z)
            res = g*(1-g)
        return res


    def cross_entropy(self, y, output):
        return -np.sum(y*np.log(output), axis=(0, 1))/(y.shape[1])


    def cost(self, y, output, cost_type, lmbd=0):
        if cost_type == 'cross_entropy':
            s = 0
            for item in self.W:
                s += np.linalg.norm(item)**2
            res = self.cross_entropy(y, output) + lmbd*s
        return res


    def forward_prop(self, X, cache):
        cache['h'][0] = X
        for i in range(len(self.W)-1):
            assert self.W[i].shape[1] == cache['h'][i].shape[0]
            assert self.W[i].shape[0] == self.b[i].shape[0]
            z = self.W[i].dot(cache['h'][i]) + self.b[i]
            cache['z'][i] = z
            cache['h'][i+1] = self.activation(z, activation_type=self.hidden_activation)
        z = self.W[-1].dot(cache['h'][-1])+self.b[-1]
        cache['z'][-1] = z

        assert len(cache['z']) == len(self.W)
        assert len(cache['h']) == len(self.W)

        output = self.activation(z, activation_type=self.output_activation)

        assert output.shape[1] == X.shape[1]
        return cache, output


    # assuming output layer already computed
    def backward_prop(self, cache, d_out):
        m = d_out.shape[1]
        dz = d_out
        db = (1./m)*np.sum(dz, axis=1, keepdims=True)
        dW = (1./m)*dz.dot(cache['h'][-1].T)
        dh = self.W[-1].T.dot(dz)
        self.b[-1] = self.b[-1] - self.lr * db
        self.W[-1] = (1 - 2 * self.lr * self.lmbd) * self.W[-1] - self.lr * dW
        for i in range(len(self.W)-2, 0, -1):
            dz = dh*self.activation_derivative(cache['z'][i], activation_type=self.hidden_activation)
            db = (1./m) * np.sum(dz, axis=1, keepdims=True)
            dW = (1./m) * dz.dot(cache['h'][i].T)
            dh = self.W[i].T.dot(dz)
            self.b[i] = self.b[i] - self.lr * db
            self.W[i] = (1 - self.lr * 2 * self.lmbd) * self.W[i] - self.lr * dW


    def predict(self, X):
        h = X
        for i in range(len(self.W) -1):
            z = self.W[i].dot(h) + self.b[i]
            h = self.activation(z, activation_type=self.hidden_activation)
        z = self.W[-1].dot(h) + self.b[-1]
        return self.activation(z, activation_type=self.output_activation)


    def fit(self, X, y, k, learning_rate=1, num_epochs=30, lmbd=0, batch_size=1000, shuffle=False):

        self.lr = learning_rate
        self.lmbd = lmbd
        self.init_parameters_normal(X.shape[0], k)
        costs = []
        idxs = np.arange(X.shape[1])
        cache = {'h': [None]*len(self.W), 'z': [None]*len(self.W)}
        for n in range(num_epochs):
            if shuffle:
                np.random.shuffle(idxs)
            for i in range(idxs.size//batch_size):
                l = i*batch_size
                r = (i+1)*batch_size
                cache, output = self.forward_prop(X[:, idxs[l:r]], cache)
                costs.append(self.cost(y[:, idxs[l:r]], output, cost_type=self.cost_function, lmbd=lmbd))
                # TODO: below is a hardcoded derivative dL/dz_out, make it generic
                d_out = (output - y[:, idxs[l:r]])
                assert d_out.shape == output.shape
                self.backward_prop(cache, d_out)
        return costs