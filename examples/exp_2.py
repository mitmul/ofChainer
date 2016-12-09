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
            func.backward(inputs, grad_outputs)
            func = func.inputs[0].creator


class Function(object):

    def __call__(self, *inputs):
        in_data = [x.data for x in inputs]
        outputs = self.forward(in_data)
        ret = [Variable(y) for y in outputs]
        for y in ret:
            y.set_creator(self)
        self.inputs = inputs
        self.outputs = ret
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        return 1 * grad_outputs

data = xp.array([0, 1, 2, 3])
x = Variable(data)

f_1 = Function()
y_1 = f_1(x)
f_2 = Function()
y_2 = f_2(y_1)

print(y_2.data)
print(y_2.creator)                                   # => f_2
print(y_2.creator.inputs[0])                         # => y_1
print(y_2.creator.inputs[0].creator)                 # => f_1
print(y_2.creator.inputs[0].creator.inputs[0])       # => x
print(y_2.creator.inputs[0].creator.inputs[0].data)  # => data

def test(*a):
    print(a)


test(1, 2, 3)
test(1)
