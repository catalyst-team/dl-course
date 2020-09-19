import numpy as np

from .engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        weights = [Value(np.random.uniform(-1, 1)) for _ in range(in_features*out_features)]
        self.weights = Tensor(weights).reshape(out_features, in_features)
        self.bias = bias
        if bias:
            self.biases = Tensor([Value(np.random.uniform(-1, 1)) for _ in range(out_features)])

    def forward(self, inp):
        """Y = W * x + b"""
        x = self.weights.dot(inp)
        if self.bias:
            x += self.biases
        return x

    def parameters(self):
        return self.weights.parameters() + self.biases.parameters()


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        mask = (inp.data > 0).astype(int)
        return inp * mask


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, inp, label):
        sum_ = 0
        inp -= inp.max()
        for c in np.exp(inp):
            sum_ = sum_ + c
        return Tensor(-inp[label] + sum_)
