import numpy as np
from model import Model
from layers import Affine, ReLuActivation
from initializers import NormalInitializer
from losses import CrossEntropyLoss, SVM, SVMMax
from utils.gradient_check import evaluate_gradient
from math_utils import relative_error

N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-3
model = Model()
model.add_layer(Affine(H, input_shape=(N, D), weight_initializer=NormalInitializer(std=std)))
model.add_layer(ReLuActivation())
model.add_layer(Affine(C, weight_initializer=NormalInitializer(std=std)))
model.initialize()

print('Testing initialization ... ')
W1_std = abs(model.layers[0].W.std() - std)
b1 = model.layers[0].b
W2_std = abs(model.layers[2].W.std() - std)
b2 = model.layers[2].b
assert W1_std < std / 10, 'First layer weights do not seem right'
assert np.all(b1 == 0), 'First layer biases do not seem right'
assert W2_std < std / 10, 'Second layer weights do not seem right'
assert np.all(b2 == 0), 'Second layer biases do not seem right'

print('Testing test-time forward pass ...')
W1 = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
b1 = np.linspace(-0.1, 0.9, num=H).reshape(1, -1)
model.layers[0].weights = np.concatenate([W1, b1])
W2 = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
b2 = np.linspace(-0.9, 0.1, num=C).reshape(1, -1)
model.layers[2].weights = np.concatenate([W2, b2])
print(model.layers[0].W.shape)
print(model.layers[0].b.shape)
print(model.layers[2].W.shape)
print(model.layers[2].b.shape)


X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.predict(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, 'Problem with test-time forward pass'

print('Testing training loss (no regularization)')
model.loss = CrossEntropyLoss()
y = np.asarray([0, 5, 1])
loss, _, loss_grad = model._forward(X, y)
correct_loss = 3.4702243556
# assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'



print('Running numeric gradient check with reg = ', 0)
grads = model._backward(loss_grad, update_weights=False)

for i in [0, 2]:
    f = lambda x: model._forward(X, y, False)[0]
    grad_num = evaluate_gradient(f, model.layers[2-i].weights)
    err, idx = relative_error(grad_num, grads[i])
    print(err, grads[i][idx], grad_num[idx])
    # print('Analytical')
    # print(grads[i])
    # print('Numerical')
    # print(grad_num)