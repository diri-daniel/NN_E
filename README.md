# NN-E

A small neural network framework built from scratch in Python — no PyTorch, no TensorFlow. Layers, forward/backward propagation, loss functions, and metrics are all implemented directly on top of NumPy, with an optional OpenCL backend for the matrix multiplication step.

```bash
pip install NN-E
```

Requires Python >= 3.11.

## Why

Built to understand what's actually happening inside a neural network, rather than to replace an existing framework. Every layer, activation, loss function, and weight-initialization scheme is implemented directly rather than pulled from a library — including a hand-written OpenCL kernel path for the forward pass. Across runs, this implementation scores in the 95–99% range on MNIST, averaging around 97%.

## Quick start

```python
import numpy as np
from NN_E import Network, Layers, NetworkType

# input layer, one hidden layer, output layer
layers = [
    Layers(784),                          # input layer (no activation needed)
    Layers(128, activation="relu"),       # hidden layer
    Layers(10, activation="SFMX"),        # output layer (softmax)
]

net = Network(layers, name="mnist-demo", type=NetworkType.Simple_Neural_Network)
net.Compile(LearningRate=0.01, metrics=["Accuracy"])

net.Fit((X_train, y_train), epochs=100, n=10, batch=1000)
net.Test((X_test, y_test))
```

## Layers

Each `Layers(size, activation, alpha, backend)` takes:

- `size` — number of neurons
- `activation` — `"relu"`, `"lrelu"` (leaky ReLU, uses `alpha`), `"sigmoid"`, or `"SFMX"` (softmax)
- `backend` — `"numpy"` (default) or `"openCl"` for the matrix multiplication step

## Weight initialization

Set via `Compile(..., weightDist=...)`:

`default`, `Xavier_Uniform`, `Xavier_Normal`, `He_Uniform`, `He_Normal`, `Lecun_Normal`, `Uniform_Small`, `Zeros`

## Metrics & loss

`Compile` accepts `Loss` (`"CCE"` or `"BCE"`) and `metrics` (`"Accuracy"`, `"Precision"`, `"Recall"`, `"F1"`). The loss is usually auto-detected from the output layer's activation function.

## Preprocessing

```python
from NN_E import Preprocessor

pre = Preprocessor(train=(X_train_raw, y_train_raw), test=(X_test_raw, y_test_raw))
pre.labels(into="onehot")
pre.features(into="normalize")
(X_train, y_train), (X_test, y_test) = pre.data()
```

## Saving and loading

```python
net.save()                 # writes weights/biases to Outputs/<type>/<name>/
net.load()                 # reloads them back into the network
model_data = net.export()  # in-memory (type, name, weights) tuple, e.g. for a database
net.import_model(model_data)
```

## Visualizing training

```python
net.plotMetrics()  # plots loss and every tracked metric with matplotlib
```

## Status

This is an active, independent project rather than a finished library — the OpenCL backend currently only supports the forward pass, and a CUDA backend is stubbed but not implemented. Issues and questions welcome.

## License

MIT