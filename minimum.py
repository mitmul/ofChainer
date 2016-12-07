#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from sklearn.datasets import fetch_mldata
import numpy
numpy.random.seed(1988)

class Linear(object):

    def __init__(self, in_sz, out_sz):
        self.W = numpy.random.randn(out_sz, in_sz) * numpy.sqrt(2. / in_sz)
        self.b = numpy.zeros((out_sz,))

    def __call__(self, x):
        self.x = x
        return x.dot(self.W.T) + self.b

    def update(self, gy, lr):
        self.W -= lr * gy.T.dot(self.x)
        self.b -= lr * gy.sum(axis=0)
        return gy.dot(self.W)

class ReLU(object):

    def __call__(self, x):
        self.x = x
        return numpy.maximum(0, x)

    def update(self, gy, lr):
        return gy * (self.x > 0)

model = [
    Linear(784, 100),
    ReLU(),
    Linear(100, 100),
    ReLU(),
    Linear(100, 10)
]

def forward(model, x):
    for layer in model:
        x = layer(x)
    return x

def update(model, gy, lr=0.0001):
    for layer in reversed(model):
        gy = layer.update(gy, lr)

def softmax_cross_entropy_gy(y, t):
    return (numpy.exp(y.T) / numpy.exp(y.T).sum(axis=0)).T - t

def accuracy(y, t):
    n_correct = numpy.sum(t[numpy.arange(len(t)), y.argmax(axis=1)])
    return n_correct / float(len(t))

mnist = fetch_mldata('MNIST original', data_home='.')
td, tl = mnist.data[:60000] / 255.0, mnist.target[:60000]
tl = numpy.array([tl == i for i in range(10)]).T.astype(numpy.int)
perm = numpy.random.permutation(len(td))
td, tl = td[perm], tl[perm]

for epoch in range(30):
    for i in range(0, len(td), 128):
        x, t = td[i:i + 128], tl[i:i + 128]
        y = forward(model, x)
        gy = softmax_cross_entropy_gy(y, t)
        update(model, gy)

vd, vl = mnist.data[60000:] / 255.0, mnist.target[60000:]
vl = numpy.array([vl == i for i in range(10)]).T.astype(numpy.int)
y = forward(model, vd)
print(accuracy(y, vl))
