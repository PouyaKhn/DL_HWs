import numpy as np
from module import Module


class Linear(Module):
    def __init__(self, name, input_dim, output_dim, l2_coef=.0):
        super(Linear, self).__init__(name)
        self.l2_coef = l2_coef  # coefficient of l2 regularization.
        self.W = np.random.randn(input_dim, output_dim)  # weights of the layer.
        self.b = np.random.randn(output_dim, )  # biases of the layer.
        self.dW = None  # gradients of loss w.r.t. the weights.
        self.db = None  # gradients of loss w.r.t. the biases.

    def forward(self, x, **kwargs):
        out = x @ self.W + self.b
        self.cache = x
        return out

    def backward(self, dout):
        x = self.cache
        dx = dout @ self.W.T
        self.dW = x.T @ dout + 2.0 * self.l2_coef * self.W
        self.db = dout.sum(axis=0)
        return dx
