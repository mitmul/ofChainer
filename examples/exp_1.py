#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Variable(object):

    def __init__(self, data):
        self.data = data
        self.creator = None

    def set_creator(self, gen_func):
        self.creator = gen_func


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
        return in_data

data = [0, 1, 2, 3]
x = Variable(data)

f_1 = Function()
y_1 = f_1(x)
f_2 = Function()
y_2 = f_2(y_1)

print(y_2.data)
print(y_2.creator)                           # => f_2
print(y_2.creator.input)                     # => y_1
print(y_2.creator.input.creator)             # => f_1
print(y_2.creator.input.creator.input)       # => x
print(y_2.creator.input.creator.input.data)  # => data
