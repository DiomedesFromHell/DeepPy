import numpy as np
from numpy.lib.stride_tricks import as_strided


# first dimension batch, steps should correspond to X ndims
def window_view(X, steps, window_size, axes=None):
    if axes is None and (X.ndim != len(steps) or X.ndim != len(window_size)):
        raise ValueError("window_view: steps and window_size len should equal to X.ndim or axes should be provided.")
    elif X.ndim != len(steps) or X.ndim != len(window_size):
        if not len(steps) == len(window_size) == len(axes):
            raise ValueError("window_view: provided axes should have the same len as steps and window_size")
    else:
        axes = [i for i in range(X.ndim)]

    base_strides = list(X.strides)
    base_shape = list(X.shape)

    for i, ax in enumerate(axes):
        base_strides[ax] *= steps[i]
        base_shape[ax] = 1 + (base_shape[ax] - window_size[i]) // steps[i]

    win_strides = [X.strides[ax] for ax in axes]
    win_shape = list(window_size)

    strides = base_strides + win_strides
    shape = base_shape + win_shape

    return as_strided(X, shape, strides)
