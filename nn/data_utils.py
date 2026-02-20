"""Data utilities for loading and preparing data.

Allowed imports: csv, math, random, and Matrix from math_utils.
"""

import csv
import math
import random
from nn.math_utils import Matrix


def load_csv(filepath: str, has_header: bool = True) -> tuple:
    """Load a CSV file and return (data, headers).
    data is a list of lists of floats.
    headers is a list of strings or None."""
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ([], None)
    if has_header:
        headers = rows[0]
        rows = rows[1:]
    else:
        headers = None
    data = []
    for row in rows:
        try:
            data.append([float(x) for x in row])
        except ValueError:
            continue
    return (data, headers)


def train_test_split(
    X: Matrix,
    y: Matrix,
    test_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    """Split data into training and test sets.
    Returns (X_train, X_test, y_train, y_test).
    Shuffle before splitting."""
    if X.rows != y.rows:
        raise ValueError(f"X.rows ({X.rows}) must equal y.rows ({y.rows})")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    n = X.rows
    indices = list(range(n))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    X_train = Matrix([X.data[i][:] for i in train_idx])
    X_test = Matrix([X.data[i][:] for i in test_idx])
    y_train = Matrix([y.data[i][:] for i in train_idx])
    y_test = Matrix([y.data[i][:] for i in test_idx])
    return (X_train, X_test, y_train, y_test)


def normalize(X: Matrix) -> tuple[Matrix, list[float], list[float]]:
    """Min-max normalize each column to [0, 1] range.
    Returns (normalized_matrix, min_values, max_values).
    min_values and max_values are lists of floats per column."""
    if X.rows == 0 or X.cols == 0:
        raise ValueError("normalize requires a non-empty matrix")
    min_vals = []
    max_vals = []
    for j in range(X.cols):
        col = [X.data[i][j] for i in range(X.rows)]
        min_vals.append(min(col))
        max_vals.append(max(col))
    result_data = []
    for i in range(X.rows):
        row = []
        for j in range(X.cols):
            lo, hi = min_vals[j], max_vals[j]
            if hi == lo:
                row.append(0.0)
            else:
                row.append((X.data[i][j] - lo) / (hi - lo))
        result_data.append(row)
    return (Matrix(result_data), min_vals, max_vals)


def standardize(X: Matrix) -> tuple[Matrix, list[float], list[float]]:
    """Z-score standardize each column (mean=0, std=1).
    Returns (standardized_matrix, means, stds)."""
    if X.rows == 0 or X.cols == 0:
        raise ValueError("standardize requires a non-empty matrix")
    means = []
    stds = []
    for j in range(X.cols):
        col = [X.data[i][j] for i in range(X.rows)]
        n = len(col)
        mu = sum(col) / n
        var = sum((x - mu) ** 2 for x in col) / n
        sigma = math.sqrt(var) if var > 0 else 1.0
        means.append(mu)
        stds.append(sigma)
    result_data = []
    for i in range(X.rows):
        row = [(X.data[i][j] - means[j]) / stds[j] for j in range(X.cols)]
        result_data.append(row)
    return (Matrix(result_data), means, stds)


def generate_spiral_data(
    samples_per_class: int = 100,
    num_classes: int = 3,
    seed: int | None = None,
) -> tuple[Matrix, list[int]]:
    """Generate a spiral dataset for multi-class classification testing.
    Returns (X, y) where X is Matrix (n x 2) and y is integer labels list.
    Uses parametric spiral equations with noise."""
    if seed is not None:
        random.seed(seed)
    X_data = []
    y_labels = []
    n = samples_per_class * num_classes
    for i in range(n):
        class_id = i % num_classes
        r = (i // num_classes + 1) / samples_per_class
        t = r * 4.0 * math.pi * (class_id + 1) / num_classes + class_id * 2.0 * math.pi / num_classes
        x = r * math.cos(t) + random.gauss(0, 0.15)
        y = r * math.sin(t) + random.gauss(0, 0.15)
        X_data.append([x, y])
        y_labels.append(class_id)
    return (Matrix(X_data), y_labels)


def generate_xor_data() -> tuple[Matrix, Matrix]:
    """Generate XOR dataset.
    Returns (X, y) where X is Matrix [[0,0],[0,1],[1,0],[1,1]]
    and y is Matrix [[0],[1],[1],[0]]."""
    X = Matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Matrix([[0.0], [1.0], [1.0], [0.0]])
    return (X, y)


def generate_circles_data(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: int | None = None,
) -> tuple[Matrix, Matrix]:
    """Generate two concentric circles for binary classification.
    Inner circle = class 0, outer circle = class 1.
    Returns (X, y)."""
    if seed is not None:
        random.seed(seed)
    half = n_samples // 2
    X_data = []
    y_data = []
    for i in range(half):
        angle = random.uniform(0, 2 * math.pi)
        r = 1.0 + random.gauss(0, noise)
        X_data.append([r * math.cos(angle), r * math.sin(angle)])
        y_data.append([0.0])
    for i in range(half, n_samples):
        angle = random.uniform(0, 2 * math.pi)
        r = 2.0 + random.gauss(0, noise)
        X_data.append([r * math.cos(angle), r * math.sin(angle)])
        y_data.append([1.0])
    return (Matrix(X_data), Matrix(y_data))


def generate_linear_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    seed: int | None = None,
) -> tuple[Matrix, Matrix]:
    """Generate linear regression data.
    y = w1*x1 + w2*x2 + ... + bias + noise
    Returns (X, y)."""
    if seed is not None:
        random.seed(seed)
    X_data = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_samples)]
    weights = [random.uniform(-1, 1) for _ in range(n_features)]
    bias = random.uniform(-0.5, 0.5)
    y_data = []
    for i in range(n_samples):
        y_val = bias + sum(X_data[i][j] * weights[j] for j in range(n_features))
        y_val += random.gauss(0, noise)
        y_data.append([y_val])
    return (Matrix(X_data), Matrix(y_data))


def accuracy_score(predictions: Matrix, targets: Matrix) -> float:
    """Calculate classification accuracy.
    For one-hot: compare argmax per row.
    For binary: round predictions to 0/1 and compare."""
    if predictions.rows != targets.rows:
        raise ValueError("predictions and targets must have same number of rows")
    if predictions.rows == 0:
        return 0.0
    if targets.cols == 1:
        correct = 0
        for i in range(predictions.rows):
            pred_label = round(predictions.data[i][0])
            pred_label = max(0, min(1, pred_label))
            true_label = round(targets.data[i][0])
            true_label = max(0, min(1, true_label))
            if pred_label == true_label:
                correct += 1
        return correct / predictions.rows
    pred_labels = predictions.max_index_per_row()
    target_labels = targets.max_index_per_row()
    correct = sum(1 for p, t in zip(pred_labels, target_labels) if p == t)
    return correct / predictions.rows


def confusion_matrix(
    predictions: Matrix,
    targets: Matrix,
    num_classes: int,
) -> list[list[int]]:
    """Compute confusion matrix as nested list.
    Returns list[list[int]] of shape (num_classes x num_classes).
    Row = true class, column = predicted class."""
    if predictions.rows != targets.rows:
        raise ValueError("predictions and targets must have same number of rows")
    pred_labels = predictions.max_index_per_row()
    target_labels = targets.max_index_per_row()
    cm = [[0] * num_classes for _ in range(num_classes)]
    for true_idx, pred_idx in zip(target_labels, pred_labels):
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            cm[true_idx][pred_idx] += 1
    return cm


def print_confusion_matrix(cm: list[list[int]], class_names: list | None = None) -> None:
    """Pretty print a confusion matrix with labels."""
    n = len(cm)
    if n == 0:
        print("(empty confusion matrix)")
        return
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    width = max(len(str(x)) for row in cm for x in row)
    width = max(width, max(len(name) for name in class_names), 2)
    header = "True \\ Pred".ljust(width + 2)
    for name in class_names:
        header += str(name).rjust(width + 1)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        label = class_names[i] if i < len(class_names) else str(i)
        line = label.ljust(width + 2)
        for val in row:
            line += str(val).rjust(width + 1)
        print(line)
