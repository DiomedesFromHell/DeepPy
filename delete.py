# import numpy as np
# import time
#
# batch_size = 1024
# w, h = 16, 16
# c = 32
# d = 64
#
# x = np.arange(batch_size * w * h * c, dtype='float32').reshape((batch_size, w, h, c)) / 10e8
#
# y = np.arange(w * h * c * d, dtype='float32').reshape((w, h, c, d))/10e5
#
# t0 = time.time()
# es = np.einsum('bijc, ijcd->bd', x, y)
# print(time.time() - t0)
# t0 = time.time()
# td = np.tensordot(x, y, axes=((1, 2, 3), (0, 1, 2)))
# print(time.time() - t0)
# print(es)
# print('--------------------')
# print(td)
# assert np.allclose(es, td)
import numpy as np

# x = np.arange(16).reshape((2, 2, 2, 2))
# print(x)
# print('---------')
# print(x[:, 0, 1, :].shape)
# print(x[:, 0, 1, :])


x = np.arange(16).reshape([4, 4])
print(x)