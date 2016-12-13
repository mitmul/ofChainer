#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as xp


class Variable(object):

    def __init__(self, data):
        self.data = data
        self.creator = None
        self.grad = 1

    def set_creator(self, gen_func):
        self.creator = gen_func

    def backward(self):
        if self.creator is None:  # input data
            return
        func = self.creator
        while func:
            gy = func.output.grad
            func.input.grad = func.backward(gy)
            func = func.input.creator


class Function(object):

    def __call__(self, in_var):
        in_data = in_var.data
        output = self.forward(in_data)
        ret = Variable(output)
        ret.set_creator(self)
        self.input = in_var
        self.output = ret
        return ret

    def forward(self, in_data):
        NotImplementedError()

    def backward(self, grad_output):
        NotImplementedError()


class Mul(Function):

    def __init__(self, init_w):
        self.w = init_w  # Initialize the parameter

    def forward(self, in_var):
        return in_var * self.w

    def backward(self, grad_output):
        gx = self.w * grad_output
        self.gw = self.input
        return gx

data = xp.array([0, 1, 2, 3])

f1 = Mul(2)
f2 = Mul(3)
f3 = Mul(4)

y0 = Variable(data)
y1 = f1(y0)          # y1 = y0 * 2
y2 = f2(y1)          # y2 = y1 * 3
y3 = f3(y2)          # y3 = y2 * 4

print(y0.data)
print(y1.data)
print(y2.data)
print(y3.data)

y3.backward()

print(y3.grad)  # df3 / dy3 = 1
print(y2.grad)  # df3 / dy2 = (df3 / dy3) * (dy3 / dy2) = 1 * 4
print(y1.grad)  # df3 / dy1 = (df3 / dy3) * (dy3 / dy2) * (dy2 / dy1) = 1 * 4 * 3
print(y0.grad)  # df3 / dy0 = (df3 / dy3) * (dy3 / dy2) * (dy2 / dy1) * (dy1 / dy0) = 1 * 4 * 3 * 2

print(f3.gw.data)
print(f2.gw.data)
print(f1.gw.data)
