"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        init_weight = init.kaiming_uniform(fan_in=in_features, fan_out=out_features)
        self.weight = Parameter(Tensor(init_weight), device=device, dtype=dtype)
        if bias:
            init_bias = ops.transpose(init.kaiming_uniform(fan_in=out_features, fan_out=1))
            self.bias = Parameter(Tensor(init_bias), device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias != None:
            y += ops.broadcast_to(self.bias, y.shape)
        
        return y
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return X.reshape((batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        label_num = logits.shape[-1]
        y = init.one_hot(label_num, y)
        logits = ops.logsumexp(logits, axes=(-1,)) - (logits * y).sum(axes=(-1, ))
        return logits.sum() / y.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        momentum = self.momentum
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        x_mean = x.sum(axes=(0, )) / batch_size
        x_minus_mean = x - x_mean.reshape((1, feature_size)).broadcast_to(x.shape)
        x_var = ((x_minus_mean ** 2).sum(axes=(0, )) / batch_size + self.eps)
        x_std = ((x_minus_mean ** 2).sum(axes=(0, )).reshape((1, feature_size))\
        / batch_size + self.eps) ** 0.5
        # running variable
        # print(self.running_mean.shape, x_mean.shape)
        # print(self.running_var.shape, x_std.shape)
        if self.training:
            self.running_mean = (1 - momentum) * self.running_mean \
                        + momentum * x_mean.data
            self.running_var = (1 - momentum) * self.running_var \
                        + momentum * x_var.data
            norm = x_minus_mean / x_std.broadcast_to(x.shape)
        else:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / (self.running_var + self.eps).broadcast_to(x.shape) ** 0.5
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim)
        bias = init.zeros(dim)
        self.weight = Parameter(Tensor(weight), device=device, dtype=dtype)
        self.bias = Parameter(Tensor(bias), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        E_x = x.sum(axes=(-1, )).reshape((x.shape[0], 1)) / x.shape[-1]
        x_minus_mean = x - E_x.broadcast_to(x.shape)
        x_std = (((x_minus_mean ** 2).sum(axes=(1, )).reshape((x.shape[0], 1))\
        / x.shape[-1] + self.eps) ** 0.5).broadcast_to(x.shape)
        # may overflow! 
        # E_x2 = (x ** 2).sum(axes=(-1, )).reshape((x.shape[0], 1)) / x.shape[-1]
        # x_std = (E_x2 - E_x ** 2 + self.eps) ** 0.5
        y = self.weight.broadcast_to(x.shape) * x_minus_mean / \
            x_std + self.bias.broadcast_to(x.shape)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=(1-self.p))
            y = mask * x / (1-self.p)
        else:
            y = x
        return y
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



