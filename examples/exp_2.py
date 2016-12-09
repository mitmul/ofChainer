#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as xp


class Variable(object):

    def __init__(self, data):
        self.data = data
        self.creator = None

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
        return in_data ** 2

    def backward(self, gy):
        gx = 2 * self.input.data
        # パラメータがある場合、ここでgxを使ったパラメータの更新を行う
        return gy * gx

data = xp.array([0, 1, 2, 3])
x = Variable(data)

f_1 = Function()
y_1 = f_1(x)       # y_1 = x^2
f_2 = Function()
y_2 = f_2(y_1)     # y_2 = (x^2)^2

print(y_1.data)
print(y_2.data)

y_2.grad = 1
y_2.backward()

print(y_1.grad)  # d y_2 / d y_1 = 2
print(x.grad)    # d y_2 / d x = 4
