# 1f-Chainer

One file Chainer

## About One-file Chainer

- Core functionalities of Chainer are implemented in 1 file
- Only Linear, ReLU, MeanSquaredError and SGD are available
- You can run "mnist.py" to make sure that it can train a 3-layers perceptron for classifying MNIST
- 1 hot feature:

  - You can switch all computations to GPU-mode by just replacing `import numpy as xp` with `import cupy as xp` found in the top of the file "chainer.py"


## MNIST Training example

### Run on CPU

- Just run `mnist.py`

```
python mnist.py
```

### Run on GPU

- Replace `import numpy as xp` with `import cupy as xp` in `chainer.py`
- Then, run `mnist.py` with an option `--gpu 0`

```
python mnist.py --gpu 0
```

## minimum.py?

"minimum.py" in this repository is another implementation example of 3-layers perceptron for MNIST. It contains

- Linear
- ReLU
- Softmax cross entropy (grad)
- Training code for MNIST
- Accuracy calculation

in **just 74 lines**. ("chainer.py" have 205 lines)

Try:

```
python minimum.py
```
