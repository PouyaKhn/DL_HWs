import numpy as np
from module import Module

class Sigmoid(Module):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of Sigmoid function for input x.
        """
        out = 1/(1+np.exp(-x))
        self.cache = x
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        x = self.cache
        s = 1/(1+np.exp(-x))
        dx = dout * s * (1-s)

        return dx
