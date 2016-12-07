from chainer import *
from sklearn.datasets import fetch_mldata

import argparse
import cupy as xp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

if __name__ == '__main__':
    mnist, t, bsize = fetch_mldata('MNIST original', data_home='.'), 60000, 32
    td, tl = mnist.data[:t].astype(np.float32) / 255.0, mnist.target[:t]
    tl = np.array([tl == i for i in range(10)]).T.astype(np.float32)
    perm = np.random.permutation(len(td))
    td, tl = td[perm], tl[perm]
    if args.gpu >= 0:
        td, tl = xp.asarray(td), xp.asarray(tl)

    model = Chain(
        l1=Linear(784, 100),
        l2=Linear(100, 100),
        l3=Linear(100, 10)
    )

    def forward(x):
        h = model.l1(x)
        h = relu(h)
        h = model.l2(h)
        h = relu(h)
        h = model.l3(h)
        return h

    opt = SGD(lr=0.1)
    opt.setup(model)

    n_iter, losses = 0, []
    for epoch in range(1):
        for i in range(0, len(td), bsize):
            x = Variable(td[i:i + bsize])
            t = Variable(tl[i:i + bsize])
            # forward
            y = forward(x)
            # backward
            loss = mean_squared_error(y, t)
            model.zerograds()
            loss.backward()
            opt.update()
            n_iter += 1
            losses.append(loss.data)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig('loss.png')
