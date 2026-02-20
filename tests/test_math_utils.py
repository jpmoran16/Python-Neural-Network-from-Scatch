"""Tests for math_utils module."""

import unittest
from nn.math_utils import Matrix, vector_to_matrix, one_hot_encode


class TestMatrixCreation(unittest.TestCase):
    """Test Matrix creation, zeros, and random constructors."""

    def test_matrix_creation_from_list(self):
        """Matrix created from nested list has correct shape and data."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        m = Matrix(data)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.data, data)
        self.assertEqual(m.shape(), (2, 2))

    def test_matrix_zeros(self):
        """Matrix.zeros creates a matrix of zeros with correct shape."""
        m = Matrix.zeros(3, 4)
        self.assertEqual(m.shape(), (3, 4))
        for i in range(3):
            for j in range(4):
                self.assertEqual(m.data[i][j], 0.0)

    def test_matrix_random_shape_and_bounds(self):
        """Matrix.random produces matrix in [low, high) with correct shape."""
        m = Matrix.random(2, 3, low=-1.0, high=1.0, seed=42)
        self.assertEqual(m.shape(), (2, 3))
        for i in range(2):
            for j in range(3):
                self.assertGreaterEqual(m.data[i][j], -1.0)
                self.assertLess(m.data[i][j], 1.0)

    def test_matrix_random_reproducible_with_seed(self):
        """Matrix.random with same seed produces same result."""
        m1 = Matrix.random(2, 2, seed=123)
        m2 = Matrix.random(2, 2, seed=123)
        self.assertEqual(m1.data, m2.data)

    def test_matrix_from_list_reshape(self):
        """Matrix.from_list reshapes flat list correctly."""
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m = Matrix.from_list(flat, 2, 3)
        self.assertEqual(m.shape(), (2, 3))
        self.assertEqual(m.data[0], [1.0, 2.0, 3.0])
        self.assertEqual(m.data[1], [4.0, 5.0, 6.0])


class TestMatrixArithmetic(unittest.TestCase):
    """Test addition, subtraction, multiplication (element-wise and scalar)."""

    def test_addition_matrix_matrix(self):
        """Element-wise Matrix + Matrix gives correct result."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a + b
        self.assertEqual(c.data, [[6.0, 8.0], [10.0, 12.0]])

    def test_addition_matrix_scalar(self):
        """Matrix + scalar adds scalar to every element."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a + 10.0
        self.assertEqual(c.data, [[11.0, 12.0], [13.0, 14.0]])

    def test_subtraction_matrix_matrix(self):
        """Element-wise Matrix - Matrix gives correct result."""
        a = Matrix([[5.0, 6.0], [7.0, 8.0]])
        b = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a - b
        self.assertEqual(c.data, [[4.0, 4.0], [4.0, 4.0]])

    def test_subtraction_matrix_scalar(self):
        """Matrix - scalar subtracts scalar from every element."""
        a = Matrix([[10.0, 20.0], [30.0, 40.0]])
        c = a - 5.0
        self.assertEqual(c.data, [[5.0, 15.0], [25.0, 35.0]])

    def test_multiplication_element_wise(self):
        """Element-wise Matrix * Matrix (Hadamard) gives correct result."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[2.0, 3.0], [4.0, 5.0]])
        c = a * b
        self.assertEqual(c.data, [[2.0, 6.0], [12.0, 20.0]])

    def test_multiplication_scalar(self):
        """Matrix * scalar multiplies every element."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        c = a * 2.0
        self.assertEqual(c.data, [[2.0, 4.0], [6.0, 8.0]])


class TestMatrixDotProduct(unittest.TestCase):
    """Test dot product with known values."""

    def test_dot_product_known_result(self):
        """[[1,2],[3,4]] dot [[5,6],[7,8]] = [[19,22],[43,50]]."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[5.0, 6.0], [7.0, 8.0]])
        c = a.dot(b)
        self.assertEqual(c.rows, 2)
        self.assertEqual(c.cols, 2)
        self.assertAlmostEqual(c.data[0][0], 19.0)
        self.assertAlmostEqual(c.data[0][1], 22.0)
        self.assertAlmostEqual(c.data[1][0], 43.0)
        self.assertAlmostEqual(c.data[1][1], 50.0)


class TestMatrixTranspose(unittest.TestCase):
    """Test transpose."""

    def test_transpose_2x3(self):
        """Transpose of 2x3 matrix is 3x2 with correct elements."""
        m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = m.transpose()
        self.assertEqual(t.shape(), (3, 2))
        self.assertEqual(t.data[0], [1.0, 4.0])
        self.assertEqual(t.data[1], [2.0, 5.0])
        self.assertEqual(t.data[2], [3.0, 6.0])


class TestMatrixApply(unittest.TestCase):
    """Test apply with function."""

    def test_apply_lambda_double(self):
        """apply with lambda x: x * 2 doubles every element."""
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        out = m.apply(lambda x: x * 2)
        self.assertEqual(out.data, [[2.0, 4.0], [6.0, 8.0]])


class TestMatrixSums(unittest.TestCase):
    """Test sum_all, sum_rows, sum_cols."""

    def test_sum_all(self):
        """sum_all returns sum of all elements."""
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(m.sum_all(), 10.0)

    def test_sum_rows(self):
        """sum_rows returns column vector of row sums."""
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        s = m.sum_rows()
        self.assertEqual(s.shape(), (2, 1))
        self.assertEqual(s.data[0][0], 3.0)
        self.assertEqual(s.data[1][0], 7.0)

    def test_sum_cols(self):
        """sum_cols returns row vector of column sums."""
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        s = m.sum_cols()
        self.assertEqual(s.shape(), (1, 2))
        self.assertEqual(s.data[0], [4.0, 6.0])


class TestMatrixMismatchedDimensions(unittest.TestCase):
    """Test that mismatched dimensions raise ValueError."""

    def test_add_mismatched_shapes_raises(self):
        """Matrix + Matrix with different shapes raises ValueError."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])
        b = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaises(ValueError) as ctx:
            a + b
        self.assertIn("shape", str(ctx.exception).lower())

    def test_dot_incompatible_dimensions_raises(self):
        """Dot product with incompatible dimensions raises ValueError."""
        a = Matrix([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        b = Matrix([[1.0], [2.0], [3.0]])     # 3x1
        with self.assertRaises(ValueError) as ctx:
            a.dot(b)
        self.assertIn("Incompatible", str(ctx.exception))


class TestOneHotEncode(unittest.TestCase):
    """Test one_hot_encode."""

    def test_one_hot_encode_three_classes(self):
        """[0,1,2] with 3 classes -> [[1,0,0],[0,1,0],[0,0,1]]."""
        result = one_hot_encode([0, 1, 2], 3)
        self.assertEqual(result.shape(), (3, 3))
        self.assertEqual(result.data[0], [1.0, 0.0, 0.0])
        self.assertEqual(result.data[1], [0.0, 1.0, 0.0])
        self.assertEqual(result.data[2], [0.0, 0.0, 1.0])


class TestVectorToMatrix(unittest.TestCase):
    """Test vector_to_matrix."""

    def test_vector_to_matrix_column(self):
        """vector_to_matrix with column=True gives column vector (n x 1)."""
        v = [1.0, 2.0, 3.0]
        m = vector_to_matrix(v, column=True)
        self.assertEqual(m.shape(), (3, 1))
        self.assertEqual(m.data[0][0], 1.0)
        self.assertEqual(m.data[1][0], 2.0)
        self.assertEqual(m.data[2][0], 3.0)

    def test_vector_to_matrix_row(self):
        """vector_to_matrix with column=False gives row vector (1 x n)."""
        v = [1.0, 2.0, 3.0]
        m = vector_to_matrix(v, column=False)
        self.assertEqual(m.shape(), (1, 3))
        self.assertEqual(m.data[0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
