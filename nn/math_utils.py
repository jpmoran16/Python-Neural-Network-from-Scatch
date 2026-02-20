"""Mathematical utilities for neural networks - Matrix operations without NumPy."""

import math
import random
from typing import Callable


class Matrix:
    """A 2D matrix class built on nested Python lists."""
    
    def __init__(self, data: list[list[float]]):
        """
        Initialize a Matrix from nested list data.
        
        Args:
            data: Nested list of floats representing matrix elements.
                  All rows must have the same length.
        
        Raises:
            ValueError: If data is empty or rows have inconsistent lengths.
        """
        if not data:
            raise ValueError("Matrix data cannot be empty")
        
        if not isinstance(data[0], list):
            raise ValueError("Matrix data must be a nested list")
        
        self.cols = len(data[0])
        if self.cols == 0:
            raise ValueError("Matrix rows cannot be empty")
        
        # Validate all rows have the same length
        for i, row in enumerate(data):
            if len(row) != self.cols:
                raise ValueError(f"Row {i} has length {len(row)}, expected {self.cols}")
        
        self.data = data
        self.rows = len(data)
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """
        Create a matrix of zeros.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
        
        Returns:
            A Matrix filled with zeros.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and cols must be positive integers")
        return Matrix([[0.0] * cols for _ in range(rows)])
    
    @staticmethod
    def random(rows: int, cols: int, low: float = -1.0, high: float = 1.0, seed: int | None = None) -> 'Matrix':
        """
        Create a matrix with random values between low and high.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
            low: Lower bound for random values (inclusive).
            high: Upper bound for random values (exclusive).
            seed: Optional random seed for reproducibility.
        
        Returns:
            A Matrix filled with random values.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and cols must be positive integers")
        if low >= high:
            raise ValueError("low must be less than high")
        
        if seed is not None:
            random.seed(seed)
        
        data = [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    
    @staticmethod
    def from_list(flat_list: list[float], rows: int, cols: int) -> 'Matrix':
        """
        Reshape a flat list into a matrix.
        
        Args:
            flat_list: Flat list of values.
            rows: Number of rows.
            cols: Number of columns.
        
        Returns:
            A Matrix with the reshaped data.
        
        Raises:
            ValueError: If flat_list length doesn't match rows * cols.
        """
        if len(flat_list) != rows * cols:
            raise ValueError(f"flat_list length {len(flat_list)} doesn't match rows*cols ({rows}*{cols}={rows*cols})")
        
        data = [flat_list[i*cols:(i+1)*cols] for i in range(rows)]
        return Matrix(data)
    
    def transpose(self) -> 'Matrix':
        """
        Return the transpose of this matrix.
        
        Returns:
            A new Matrix that is the transpose of this matrix.
        """
        if self.rows == 0 or self.cols == 0:
            return Matrix([])
        
        transposed_data = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(transposed_data)
    
    def __add__(self, other) -> 'Matrix':
        """
        Element-wise addition. Support Matrix + Matrix and Matrix + scalar.
        
        Args:
            other: Either a Matrix (same shape) or a scalar (int/float).
        
        Returns:
            A new Matrix with the result.
        
        Raises:
            ValueError: If shapes don't match for Matrix addition.
            TypeError: If other is not a Matrix or scalar.
        """
        if isinstance(other, (int, float)):
            result_data = [[self.data[i][j] + other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        elif isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError(f"Matrix shapes don't match: ({self.rows}, {self.cols}) vs ({other.rows}, {other.cols})")
            result_data = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError(f"Unsupported operand type for +: Matrix and {type(other)}")
    
    def __radd__(self, other) -> 'Matrix':
        """Right addition (scalar + Matrix)."""
        return self.__add__(other)
    
    def __sub__(self, other) -> 'Matrix':
        """
        Element-wise subtraction. Support Matrix - Matrix and Matrix - scalar.
        
        Args:
            other: Either a Matrix (same shape) or a scalar (int/float).
        
        Returns:
            A new Matrix with the result.
        
        Raises:
            ValueError: If shapes don't match for Matrix subtraction.
            TypeError: If other is not a Matrix or scalar.
        """
        if isinstance(other, (int, float)):
            result_data = [[self.data[i][j] - other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        elif isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError(f"Matrix shapes don't match: ({self.rows}, {self.cols}) vs ({other.rows}, {other.cols})")
            result_data = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError(f"Unsupported operand type for -: Matrix and {type(other)}")
    
    def __rsub__(self, other) -> 'Matrix':
        """Right subtraction (scalar - Matrix)."""
        if isinstance(other, (int, float)):
            result_data = [[other - self.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError(f"Unsupported operand type for -: {type(other)} and Matrix")
    
    def __mul__(self, other) -> 'Matrix':
        """
        Element-wise multiplication (Hadamard product) for Matrix * Matrix.
        Scalar multiplication for Matrix * float/int.
        
        Args:
            other: Either a Matrix (same shape) or a scalar (int/float).
        
        Returns:
            A new Matrix with the result.
        
        Raises:
            ValueError: If shapes don't match for Matrix multiplication.
            TypeError: If other is not a Matrix or scalar.
        """
        if isinstance(other, (int, float)):
            result_data = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        elif isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError(f"Matrix shapes don't match: ({self.rows}, {self.cols}) vs ({other.rows}, {other.cols})")
            result_data = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError(f"Unsupported operand type for *: Matrix and {type(other)}")
    
    def __rmul__(self, other) -> 'Matrix':
        """Right multiplication (scalar * Matrix)."""
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'Matrix':
        """
        Element-wise division by scalar.
        
        Args:
            other: A scalar (int/float).
        
        Returns:
            A new Matrix with the result.
        
        Raises:
            ZeroDivisionError: If other is zero.
            TypeError: If other is not a scalar.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            result_data = [[self.data[i][j] / other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(result_data)
        else:
            raise TypeError(f"Unsupported operand type for /: Matrix and {type(other)}")
    
    def dot(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix multiplication (dot product).
        self is (m x n), other is (n x p), result is (m x p).
        
        Args:
            other: A Matrix with compatible dimensions.
        
        Returns:
            A new Matrix that is the matrix product.
        
        Raises:
            ValueError: If matrix dimensions are incompatible.
        """
        if self.cols != other.rows:
            raise ValueError(f"Incompatible matrix dimensions for dot product: ({self.rows}, {self.cols}) dot ({other.rows}, {other.cols})")
        
        result_data = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                dot_product = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                row.append(dot_product)
            result_data.append(row)
        
        return Matrix(result_data)
    
    def apply(self, func: Callable[[float], float]) -> 'Matrix':
        """
        Apply a function element-wise and return new Matrix.
        
        Args:
            func: A function that takes a float and returns a float.
        
        Returns:
            A new Matrix with the function applied to each element.
        """
        result_data = [[func(self.data[i][j]) for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result_data)
    
    def sum_all(self) -> float:
        """
        Sum all elements in the matrix.
        
        Returns:
            The sum of all elements.
        """
        return sum(sum(row) for row in self.data)
    
    def sum_rows(self) -> 'Matrix':
        """
        Sum across rows, returning a column vector (rows x 1).
        
        Returns:
            A column vector Matrix with row sums.
        """
        sums = [sum(row) for row in self.data]
        return Matrix([[s] for s in sums])
    
    def sum_cols(self) -> 'Matrix':
        """
        Sum across columns, returning a row vector (1 x cols).
        
        Returns:
            A row vector Matrix with column sums.
        """
        if self.rows == 0:
            return Matrix([[]])
        sums = [sum(self.data[i][j] for i in range(self.rows)) for j in range(self.cols)]
        return Matrix([sums])
    
    def max_index_per_row(self) -> list[int]:
        """
        Return the column index of the max value in each row.
        
        Returns:
            A list of column indices, one per row.
        """
        if self.rows == 0:
            return []
        
        indices = []
        for row in self.data:
            max_val = row[0]
            max_idx = 0
            for j in range(1, len(row)):
                if row[j] > max_val:
                    max_val = row[j]
                    max_idx = j
            indices.append(max_idx)
        
        return indices
    
    def shape(self) -> tuple[int, int]:
        """
        Return (rows, cols).
        
        Returns:
            A tuple (rows, cols) representing the matrix dimensions.
        """
        return (self.rows, self.cols)
    
    def __repr__(self) -> str:
        """
        Pretty print the matrix.
        
        Returns:
            A string representation of the matrix.
        """
        if self.rows == 0 or self.cols == 0:
            return "Matrix([])"
        
        # Format each row
        rows_str = []
        for row in self.data:
            row_str = "[" + ", ".join(f"{val:.4f}" if isinstance(val, float) else str(val) for val in row) + "]"
            rows_str.append(row_str)
        
        return "Matrix([\n  " + ",\n  ".join(rows_str) + "\n])"
    
    def to_list(self) -> list[list[float]]:
        """
        Return the raw nested list.
        
        Returns:
            A deep copy of the nested list data.
        """
        return [row[:] for row in self.data]
    
    def copy(self) -> 'Matrix':
        """
        Return a deep copy.
        
        Returns:
            A new Matrix with copied data.
        """
        return Matrix(self.to_list())
    
    def clip(self, min_val: float, max_val: float) -> 'Matrix':
        """
        Clip all values between min and max.
        
        Args:
            min_val: Minimum value to clip to.
            max_val: Maximum value to clip to.
        
        Returns:
            A new Matrix with clipped values.
        
        Raises:
            ValueError: If min_val > max_val.
        """
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
        
        result_data = []
        for row in self.data:
            clipped_row = [max(min_val, min(max_val, val)) for val in row]
            result_data.append(clipped_row)
        
        return Matrix(result_data)


def vector_to_matrix(vec: list[float], column: bool = True) -> Matrix:
    """
    Convert a 1D list to a Matrix. If column=True, returns (n x 1).
    If column=False, returns (1 x n).
    
    Args:
        vec: A 1D list of floats.
        column: If True, return a column vector (n x 1). If False, return a row vector (1 x n).
    
    Returns:
        A Matrix representing the vector.
    
    Raises:
        ValueError: If vec is empty.
    """
    if not vec:
        raise ValueError("Vector cannot be empty")
    
    if column:
        # Return column vector (n x 1)
        return Matrix([[v] for v in vec])
    else:
        # Return row vector (1 x n)
        return Matrix([vec])


def one_hot_encode(labels: list[int], num_classes: int) -> Matrix:
    """
    Convert integer labels to one-hot encoded Matrix.
    Returns Matrix of shape (len(labels) x num_classes).
    
    Args:
        labels: List of integer labels (0-indexed).
        num_classes: Total number of classes.
    
    Returns:
        A Matrix where each row is a one-hot encoded vector.
    
    Raises:
        ValueError: If any label is out of range [0, num_classes) or num_classes <= 0.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    
    if not labels:
        return Matrix([])
    
    # Validate labels
    for i, label in enumerate(labels):
        if not isinstance(label, int):
            raise ValueError(f"Label at index {i} is not an integer: {label}")
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} at index {i} is out of range [0, {num_classes})")
    
    # Create one-hot encoding
    encoded_data = []
    for label in labels:
        row = [0.0] * num_classes
        row[label] = 1.0
        encoded_data.append(row)
    
    return Matrix(encoded_data)
