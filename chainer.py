import numpy as xp
import heapq

class Variable(object):

    def __init__(self, data, grad=None, name=None):
        self.data = data
        self.rank = 0
        self.grad = grad
        self.creator = None
        self.name = name

    def set_creator(self, gen_func):
        self.creator = gen_func
        self.rank = gen_func.rank + 1

    def backward(self):
        if self.creator is None:
            return
        if self.data.size == 1 and self.grad is None:  # Loss variable
            self.grad = xp.ones_like(self.data)

        cand_funcs = []
        seen_set = set()
        seen_vars = set()
        need_copy = set()

        def add_cand(cand):
            if cand not in seen_set:
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            in_data = [x.data for x in func.inputs]
            out_grad = [y.grad for y in func.outputs]
            gxs = func.backward(in_data, out_grad)
            for x, gx in zip(func.inputs, gxs):
                if gx is None:
                    continue
                id_x = id(x)
                if x.creator is None:  # leaf
                    if x.grad is None:
                        x.grad = gx
                        need_copy.add(id_x)
                    elif id_x in need_copy:
                        x.grad = x.grad + gx
                        need_copy.remove(id_x)
                    else:
                        x.grad += gx
                else:  # not a leaf
                    add_cand(x.creator)
                    if id_x not in seen_vars:
                        x.grad = gx
                        seen_vars.add(id_x)
                        need_copy.add(id_x)
                    elif id_x in need_copy:
                        x.grad = gx + x.grad
                        need_copy.remove(id_x)
                    else:
                        x.grad += gx

    def zerograd(self):
        self.grad.fill(0)

class Function(object):

    def __call__(self, *inputs):
        in_data = [x.data for x in inputs]
        outputs = self.forward(in_data)
        ret = [Variable(y) for y in outputs]
        self.rank = max([x.rank for x in inputs])
        for y in ret:
            y.set_creator(self)
        self.inputs = inputs
        self.outputs = ret
        return ret if len(ret) > 1 else ret[0]

    def forward(self, inputs):
        NotImplementedError()

    def backward(self, inputs, grad_outputs):
        NotImplementedError()

class Link(object):

    def __init__(self, **params):
        for name, value in params.items():
            grad = xp.full_like(value, 0)
            var = Variable(value, grad, name)
            self.__dict__[name] = var

    def params(self):
        for param in self.__dict__.values():
            yield param

    def namedparams(self):
        for name, param in self.__dict__.items():
            yield '/' + name, param

    def zerograds(self):
        for param in self.params():
            param.zerograd()

class Chain(Link):

    def __init__(self, **links):
        super(Chain, self).__init__()
        self.children = []
        for name, link in links.items():
            self.children.append(name)
            self.__dict__[name] = link

    def params(self):
        for name in self.children:
            for param in self.__dict__[name].params():
                yield param

    def namedparams(self):
        for name in self.children:
            prefix = '/' + name
            for path, param in self.__dict__[name].namedparams():
                yield prefix + path, param

class Linear(Link):

    def __init__(self, in_size, out_size):
        n = xp.random.normal
        scale = xp.sqrt(2. / in_size)
        W = n(loc=0.0, scale=scale, size=(out_size, in_size))
        b = n(loc=0.0, scale=scale, size=(out_size,))
        super(Linear, self).__init__(
            W=W.astype(xp.float32), b=b.astype(xp.float32))

    def __call__(self, x):
        return LinearFunction()(x, self.W, self.b)

class LinearFunction(Function):

    def forward(self, inputs):
        x, W, b = inputs
        return x.dot(W.T) + b,

    def backward(self, inputs, grad_outputs):
        x, W, b = inputs
        gy = grad_outputs[0]
        gx = gy.dot(W).reshape(x.shape)
        gW = gy.T.dot(x)
        gb = gy.sum(0)
        return gx, gW, gb

class ReLU(Function):

    def forward(self, inputs):
        return xp.maximum(inputs[0], 0),

    def backward(self, inputs, grad_outputs):
        return grad_outputs[0] * (inputs[0] > 0),

def relu(x):
    return ReLU()(x)

class MeanSquaredError(Function):

    def forward(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.size,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        coeff = gy * (2. / self.diff.size)
        gx0 = coeff * self.diff
        return gx0, -gx0

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

class Optimizer(object):

    def setup(self, link):
        self.target = link
        self.states = {}
        self.prepare()

    def prepare(self):
        for name, param in self.target.namedparams():
            if name not in self.states:
                self.states[name] = {}

    def update(self):
        self.prepare()
        for name, param in self.target.namedparams():
            self.update_one(param, self.states[name])

class SGD(Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param, state):
        param.data -= self.lr * param.grad
