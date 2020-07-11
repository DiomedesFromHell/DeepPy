import numpy as np
import math


# df - upstream gradient
def evaluate_gradient(f, X, df=None, eps=0.000001):
    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    grad = np.zeros(X.shape)
    while not it.finished:
        idx = it.multi_index
        x0 = X[idx]
        X[idx] = x0 + eps
        f_right = f(X)
        X[idx] = x0 - eps
        f_left = f(X)
        diff = (f_right - f_left) / (2. * eps)
        if df:
            diff = df * diff
        grad[idx] = np.sum(diff)
        X[idx] = x0
        it.iternext()
    return grad


def main():
    W = np.arange(6, dtype=float).reshape(3, 2)
    a = np.array([-1., 2.5, 0.7]).reshape(1, -1)
    b = np.array([-0.3, 8.74]).reshape(-1, 1)

    f = lambda x: math.sin(a @ x @ b)
    print(evaluate_gradient(f, W))

    print('---------------')
    print(math.cos(a @ W @ b) * a.T @ b.T)


if __name__ == '__main__':
    main()
