import pytest
import numpy as np
from layers import Affine


def test_forward_backward():
    aff_layer1x1 = Affine(input_shape=(3, 1), output_size=1)
    aff_layer1x1.W = np.array([2]).reshape((1, 1))
    aff_layer1x1.b = np.zeros((1, 1))
    x = np.arange(3).reshape((3, 1))
    y1 = aff_layer1x1.forward(x)
    assert np.all(y1 == 2 * x)

    dW, db, dx = aff_layer1x1.backward(np.ones((3, 1)))
    assert np.all(dW == np.array([3]).reshape((1, 1)))
    assert np.all(db == np.array([3]).reshape((1, 1)))
    assert np.all(dx == np.array([2, 2, 2]).reshape(3, 1))

    input_shape = (1, 3)
    aff_layer = Affine(input_shape=input_shape, output_size=3)
    aff_layer.W = np.eye(3)
    aff_layer.b = np.zeros(input_shape)
    x = np.array([1, 1, 1]).reshape(input_shape)
    y = aff_layer.forward(x)
    assert np.all(y == x)

    aff_layer.W = np.arange(1, 10).reshape((3, 3)).T
    y = aff_layer.forward(x)
    assert np.all(y == np.array([6, 15, 24]).reshape(1, 3))
