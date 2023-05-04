import numpy as np
from module import Module


class ReLU(Module):
    def __init__(self, name):
        super(ReLU, self).__init__(name)

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of ReLU function for input x.
        """
        out = np.maximum(0, x)
        self.cache = x
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        x = self.cache
        dx = np.array(dout, copy=True)
        dx[x <= 0] = 0
        return dx
