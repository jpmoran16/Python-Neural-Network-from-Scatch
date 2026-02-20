# 🧠 Neural Network From Scratch

## 📃 Overview

A complete neural network library built entirely from scratch in Python using only standard library modules (`math`, `random`, `csv`, `json`). No NumPy, no TensorFlow, no PyTorch — every matrix operation, activation function, backpropagation step, and optimizer is implemented by hand.


## ❓ Why From Scratch?

Building without libraries forces you to understand every gradient, every matrix multiplication, and every weight update. You can't hide behind a framework: you have to implement the chain rule, shape your tensors correctly, and debug numerical issues yourself. It's the difference between using a car and building an engine — once you've built it, you truly know how it runs.

## ☀️ Features

- **Custom Matrix class** with full linear algebra support (dot product, transpose, element-wise ops, apply, sums)
- **Activation functions:** Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, Linear
- **Loss functions:** MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers:** SGD (with momentum), Adam, RMSProp
- **Dense (fully connected) layers** with Xavier and He initialization
- **Mini-batch gradient descent** support with optional shuffling
- **Model save/load** via JSON serialization
- **Data utilities:** normalization, standardization, train/test split, synthetic data generators (XOR, circles, spiral, linear)
- **Full test suite** with unit and integration tests (unittest)

## 🏛️ Project Structure

```
neural-network-from-scratch/
├── nn/                          # Core neural network package
│   ├── __init__.py              # Package exports (Matrix, activations, losses, layers, network, optimizers)
│   ├── math_utils.py            # Matrix class and helpers: zeros, random, dot, transpose, apply, one_hot_encode, vector_to_matrix
│   ├── activations.py           # Activation base + Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, Linear (forward + derivative)
│   ├── losses.py                # Loss base + MSE, BinaryCrossEntropy, CategoricalCrossEntropy (forward + backward)
│   ├── layers.py                # Dense layer: forward, backward (returns gradients), get/set_weights
│   ├── network.py               # Sequential Network: add, forward, backward, train, predict, evaluate, summary, save, load
│   ├── optimizers.py            # Optimizer base + SGD (momentum), Adam, RMSProp (step applies updates to layer)
│   └── data_utils.py            # load_csv, train_test_split, normalize, standardize, spiral/circles/XOR/linear generators, accuracy_score, confusion_matrix
├── tests/                       # Test suite (unittest)
│   ├── __init__.py              # Marks tests as a package
│   ├── test_math_utils.py       # Matrix creation, arithmetic, dot, transpose, apply, sums, one_hot_encode, vector_to_matrix
│   ├── test_activations.py      # Sigmoid, ReLU, Tanh, LeakyReLU, Softmax, Linear forward and derivative
│   ├── test_losses.py           # MSE, BCE, CCE forward/backward and gradient shapes
│   ├── test_layers.py           # Dense forward shape, known weights, backward shapes, Xavier/He init
│   ├── test_network.py          # add, summary, forward shape, save/load round-trip, training reduces loss
│   └── test_integration.py      # End-to-end: XOR, circles binary, spiral multiclass, Adam vs SGD
├── examples/                    # Runnable example scripts
│   ├── xor_problem.py           # XOR with MSE, 2→4→1 Sigmoid, 5000 epochs
│   ├── binary_classification.py # Circles data, train/test split, 2→8→4→1 ReLU+Sigmoid, BCE
│   └── multiclass_classification.py  # Spiral data, one-hot, 2→16→8→3 ReLU+Softmax, Adam, CCE
├── README.md                    # This file
└── .gitignore                   # Python cache, venv, IDE, OS files
```

## ▶️ Quick Start

Run from the project root (`neural-network-from-scratch/`). No installation required — Python 3.9+ and the standard library only.

### ⭕➕ XOR Problem Demo

```python
from nn import Network, Dense, Sigmoid, MSE
from nn.data_utils import generate_xor_data

X, y = generate_xor_data()

net = Network()
net.add(Dense(2, 4, Sigmoid(), weight_init="xavier"))
net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
net.set_loss(MSE())

net.train(X, y, epochs=5000, learning_rate=0.5, verbose=True, print_every=1000)

pred = net.predict(X)
print("Predictions:", [round(pred.data[i][0], 2) for i in range(4)])  # Expect ~[0, 1, 1, 0]
```

### 🖥️ Binary Classification

```python
from nn import Network, Dense, ReLU, Sigmoid, BinaryCrossEntropy
from nn.data_utils import generate_circles_data, train_test_split, accuracy_score

X, y = generate_circles_data(n_samples=200, noise=0.1, seed=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

net = Network()
net.add(Dense(2, 8, ReLU(), weight_init="he"))
net.add(Dense(8, 4, ReLU(), weight_init="he"))
net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
net.set_loss(BinaryCrossEntropy())

net.train(X_train, y_train, epochs=1000, learning_rate=0.1, verbose=True, print_every=200)

acc = accuracy_score(net.predict(X_test), y_test)
print(f"Test accuracy: {acc:.2%}")
```

### 🫏 Multi-class Classification

```python
from nn import Network, Dense, ReLU, Softmax, CategoricalCrossEntropy, one_hot_encode
from nn.optimizers import Adam
from nn.data_utils import generate_spiral_data, accuracy_score

X, y_labels = generate_spiral_data(samples_per_class=100, num_classes=3, seed=42)
y = one_hot_encode(y_labels, 3)

net = Network()
net.add(Dense(2, 16, ReLU(), weight_init="he"))
net.add(Dense(16, 8, ReLU(), weight_init="he"))
net.add(Dense(8, 3, Softmax(), weight_init="xavier"))
net.set_loss(CategoricalCrossEntropy())

net.train(X, y, epochs=3000, learning_rate=0.1,
          optimizer=Adam(learning_rate=0.01), verbose=True, print_every=500)

acc = accuracy_score(net.predict(X), y)
print(f"Accuracy: {acc:.2%}")
```

## 🥳 Running Examples

```bash
cd neural-network-from-scratch
python examples/xor_problem.py
python examples/binary_classification.py
python examples/multiclass_classification.py
```

## 📜 Running Tests

```bash
cd neural-network-from-scratch
python -m unittest discover tests/ -v
```

Or run the full suite with a pass/fail summary:

```bash
python -m tests.test_integration
```

With pytest (if installed):

```bash
python -m pytest tests/ -v
```

## 🐻‍❄️ Architecture Deep Dive

### 🔢 Forward Pass

Data flows through each layer as: **input → linear transform → activation**.

```
  Input X (batch × features)
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Layer 1:  z = X @ W1 + b1              │
  │            a = activation(z)             │
  └─────────────────────────────────────────┘
       │  a (batch × hidden)
       ▼
  ┌─────────────────────────────────────────┐
  │  Layer 2:  z = a @ W2 + b2              │
  │            a = activation(z)             │
  └─────────────────────────────────────────┘
       │  ...
       ▼
  Output (batch × output_size)
```

Each layer stores `input`, `z` (pre-activation), and `a` (post-activation) for the backward pass.

### 🦝 Backpropagation

Gradients flow backward using the chain rule. For each layer (from output back to input):

- **Output gradient:** `dL/da` — comes from the loss (e.g. `predictions - targets` for softmax + cross-entropy).
- **Pre-activation gradient:** `dL/dz = dL/da * da/dz` — element-wise product with the activation derivative (e.g. sigmoid: `a * (1 - a)`).
- **Weight gradient:** `dL/dw = input.T @ dL/dz` — how much each weight contributed to the loss.
- **Bias gradient:** `dL/db = sum(dL/dz)` over the batch (column sum).
- **Input gradient (to previous layer):** `dL/d(input) = dL/dz @ w.T` — passed as the next layer’s `dL/da`.

The optimizer then uses `dL/dw` and `dL/db` to update weights and biases (e.g. `w -= lr * dL/dw` for SGD).

### ⚖️ Weight Initialization

- **Xavier (Glorot):** scale = `sqrt(2 / (fan_in + fan_out))`. Use for **Sigmoid** and **Tanh** so activations don’t saturate.
- **He:** scale = `sqrt(2 / fan_in)`. Use for **ReLU** to keep variance of activations stable and avoid dead neurons.

Both draw weights in `[-scale, scale]` (uniform). Biases are initialized to zero.

### Optimizers Compared

| Optimizer       | Best for           | Key idea                                              |
|----------------|--------------------|--------------------------------------------------------|
| SGD            | Simple problems    | `w -= lr * gradient`                                  |
| SGD + Momentum | Faster convergence | `v = momentum * v - lr * gradient`, `w += v`           |
| Adam           | Most problems      | Per-parameter adaptive lr via biased moment estimates  |
| RMSProp        | Non-stationary     | Per-parameter scale from decaying squared gradients    |

## 🦵🏿 Limitations

- **Pure Python** — no C/BLAS, so large matrices and big datasets are slow.
- **No GPU** — everything runs on CPU.
- **No conv/recurrent layers** — only dense (fully connected) layers.
- **No batch norm or dropout** — no built-in regularization beyond the loss.
- **No sparse or specialized kernels** — matrix ops are straightforward O(n³) style matmuls.

## 🏫 What I Learned

- How **backpropagation** actually computes gradients layer by layer with the chain rule and why we need to cache activations.
- Why **weight initialization** matters: bad scale leads to vanishing/exploding activations and slow or broken training.
- How **Adam** adapts learning rates per parameter using first and second moment estimates and bias correction.
- The importance of **numerical stability**: clipping (e.g. sigmoid/softmax inputs), avoiding log(0), and preventing overflow in exponentials.
- Why **mini-batch training** often works better than full-batch: noisier updates can help escape bad minima and use data more efficiently.

## 😶‍🌫️ Dependencies

**Python 3.9+** — standard library only. No external packages (no NumPy, SciPy, sklearn, TensorFlow, PyTorch, etc.).

Allowed imports in `nn/`: `math`, `random`, `csv` (in `data_utils`), `json` (save/load), and `copy` if needed.

## License

MIT
