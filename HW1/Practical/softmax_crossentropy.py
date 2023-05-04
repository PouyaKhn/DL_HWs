import numpy as np
from module import Module

class SoftmaxCrossentropy(Module):
    def __init__(self, name):
        super(SoftmaxCrossentropy, self).__init__(name)
        self.cache = {}

    def forward(self, x, **kwargs):
        y = kwargs.pop('y', None)
        """
        x: input array.
        y: real labels for this input.
        probs: probabilities of labels for this input.
        loss: cross entropy loss between probs and real labels.
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        m = y.shape[0]
        log_likelihood = -np.log(probs[range(m), y] + 1e-8)
        loss = np.sum(log_likelihood) / m
        self.cache['probs'] = probs
        self.cache['y'] = y
        return loss, probs

    def backward(self, dout=0):
        y = self.cache['y']
        dx = self.cache['probs']
        m = y.shape[0]
        dx[range(m), y] -= 1
        dx = dx / m
        return dx
