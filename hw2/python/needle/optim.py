"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # fuck this precision!!!
            # print(ndl.Tensor(param.grad, dtype='float32').data - param.grad.data) = FALSE????
            # print(param.grad.dtype, param.grad.data.dtype)
            # param.grad.data += self.weight_decay * param.data
            grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data
            if param not in self.u:
                self.u[param] = ndl.Tensor((1-self.momentum)*grad, dtype='float32', requires_grad=False)
            else:
                self.u[param] = ndl.Tensor(self.momentum*self.u[param]+ (1-self.momentum)*grad, dtype='float32', requires_grad=False)
            # print(param.data.dtype, self.u[param].dtype)
            param.data = param.data - self.lr*self.u[param].data
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        beta1 = self.beta1
        beta2 = self.beta2
        self.t += 1
        for param in self.params:
            grad = ndl.Tensor(param.grad, dtype='float32') + self.weight_decay * param.data
            if self.t == 0 or param not in self.m or param not in self.v:
                self.m[param] = ndl.Tensor((1-beta1)*grad, dtype='float32', )
                self.v[param] = ndl.Tensor((1-beta2)*grad**2, dtype='float32', )
            else:
                self.m[param] = ndl.Tensor(beta1 * self.m[param] + (1-beta1)*grad, dtype='float32',)
                self.v[param] = ndl.Tensor(beta2 * self.v[param] + (1-beta2)*grad**2, dtype='float32',)
            # unbiased correction
            m_bar = self.m[param] / (1-beta1 ** self.t)
            v_bar = self.v[param] / (1-beta2 ** self.t)
            param.data = param.data - self.lr * m_bar.data / (self.eps + v_bar.data ** 0.5)
        return 
        ### END YOUR SOLUTION
