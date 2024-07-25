import numpy as np
from DBM import DiagonalBlockMatrix

# TEST SUIT 1: Test against expected results
# ==========================================
blocks1 = [
    [np.array([1, 0]), np.array([3, 0])],
    [np.array([4, 0]), np.array([2, 0])]
]

blocks2 = [
    [np.array([3, 0]), np.array([1, 0])],
    [np.array([2, 0]), np.array([4, 0])]
]

matrix1 = DiagonalBlockMatrix(blocks1)
matrix2 = DiagonalBlockMatrix(blocks2)

expected_blocks_add = [
    [np.array([4, 0]), np.array([4, 0])],
    [np.array([6, 0]), np.array([6, 0])]
]

expected_blocks_mult = [
    [np.array([9, 0]), np.array([13, 0])],
    [np.array([16, 0]), np.array([12, 0])]
]

# Addition test

expected_matrix_add = DiagonalBlockMatrix(expected_blocks_add)

result_matrix_add = matrix1.add(matrix2)

assert np.array_equal(result_matrix_add.to_dense(), expected_matrix_add.to_dense()), "Test failed: Addition result is incorrect."

# Multiplication test

expected_matrix_mult = DiagonalBlockMatrix(expected_blocks_mult)

result_matrix_mult = matrix1.multiply(matrix2)

assert np.array_equal(result_matrix_mult.to_dense(), expected_matrix_mult.to_dense()), "Test failed: Multiplication result is incorrect."


# TEST SUIT 2: Test against NumPy
# ==========================================
diag_blocks1 = [
    [np.array([1, 1, 1]), np.array([5, 5, 5]), np.array([0, 0, 0]), np.array([0, 0, 0])],
    [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([3, 3, 3]), np.array([0, 0, 0])],
    [np.array([0, 0, 0]), np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 0, 0])],
    [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([5, 5, 5]), np.array([4, 4, 4])]
]

diag_blocks2 = [
    [np.array([5, 5, 5]), np.array([0, 0, 0]), np.array([5, 5, 5]), np.array([0, 0, 0])],
    [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([7, 7, 7]), np.array([0, 0, 0])],
    [np.array([5, 5, 5]), np.array([6, 6, 6]), np.array([0, 0, 0]), np.array([0, 0, 0])],
    [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([8, 8, 8])]
]

matrix1 = DiagonalBlockMatrix(diag_blocks1)
matrix2 = DiagonalBlockMatrix(diag_blocks2)

# Multiplication test

result_dense_multiply = matrix1.multiply(matrix2).to_dense()

result_numpy_multiply = np.matmul(matrix1.to_dense(), matrix2.to_dense())

assert np.array_equal(result_dense_multiply, result_numpy_multiply)

# Inversion test
blocks1 = [
    [np.array([1, 2]), np.array([2, 2])],
    [np.array([2, 2]), np.array([3, 4])]
]

dense1 = np.array([[1,0,2,0],
                  [0,2,0,2],
                  [2,0,3,0],
                  [0,2,0,4]])

matrix1 = DiagonalBlockMatrix(blocks1)
inverted_custom_matrix1 = matrix1.invert()
inverted_dense_matrix1 = np.linalg.inv(dense1)

assert np.array_equal(inverted_custom_matrix1, inverted_dense_matrix1)

blocks2 = [
    [np.array([1, 2]), np.array([0, 0])],
    [np.array([0, 0]), np.array([3, 4])]
]

dense2 = np.array([[1,0,0,0],
                  [0,2,0,0],
                  [0,0,3,0],
                  [0,0,0,4]])

matrix2 = DiagonalBlockMatrix(blocks2)

inverted_custom_matrix2 = matrix2.invert().to_dense()

inverted_dense_matrix2 = np.linalg.inv(dense2)

assert np.array_equal(inverted_custom_matrix2, inverted_dense_matrix2)
