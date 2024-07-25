import numpy as np

class DiagonalBlockMatrix:
    """
    A class to represent a block matrix with diagonal blocks.

    Attributes:
        diagonal_blocks (list of list of np.array): A 2D list where each element is a diagonal matrix represented
        as a numpy array.
        block_rows (int): The number of block rows in the matrix.
        block_cols (int): The number of block columns in the matrix.
        block_size (int): The size of the diagonal blocks (assuming all blocks are of the same size).
        shape (tuple): The shape of the dense matrix representation of the block matrix.
    """
    def __init__(self, diagonal_blocks):
        """
        Initializes the DiagonalBlockMatrix with given diagonal blocks.

        Parameters:
            diagonal_blocks (list of list of np.array): A 2D list where each inner list contains diagonal matrices
            as numpy arrays.
        """
        self.diagonal_blocks = diagonal_blocks
        self.block_rows = len(diagonal_blocks)
        self.block_cols = len(diagonal_blocks[0])
        self.block_size = len(diagonal_blocks[0][0])
        self.shape = (self.block_rows * self.block_size, self.block_cols * self.block_size)

    def to_dense(self):
        """
        Converts the block matrix to its dense matrix representation.

        Returns:
            np.array: The dense matrix representation of the block matrix.
        """
        dense_matrix = np.zeros(self.shape)
        for i in range(self.block_rows):
            for j in range(self.block_cols):
                diag_elements = self.diagonal_blocks[i][j]
                start_row = i * self.block_size
                end_row = start_row + self.block_size
                start_col = j * self.block_size
                end_col = start_col + self.block_size
                if np.any(diag_elements):
                    np.fill_diagonal(dense_matrix[start_row:end_row, start_col:end_col], diag_elements)
        return dense_matrix

    def add(self, other):
        """
        Adds the current block matrix with another block matrix.

        Parameters:
            other (DiagonalBlockMatrix): The other block matrix to add.

        Returns:
            DiagonalBlockMatrix: A new DiagonalBlockMatrix representing the sum of the current and other matrices.

        Raises:
            AssertionError: If the dimensions of the current and other matrices do not match.
        """
        assert self.block_rows == other.block_rows and self.block_cols == other.block_cols, "Block matrix dimensions must match"
        result_blocks = [
            [self.diagonal_blocks[i][j] + other.diagonal_blocks[i][j] for j in range(self.block_cols)]
            for i in range(self.block_rows)
        ]
        return DiagonalBlockMatrix(result_blocks)
    
    def multiply(self, other):
        """
        Multiplies the current block matrix with another block matrix.

        Parameters:
            other (DiagonalBlockMatrix): The other block matrix to multiply with.

        Returns:
            DiagonalBlockMatrix: A new DiagonalBlockMatrix representing the product of the current and other matrices.

        Raises:
            AssertionError: If the number of columns in the current matrix does not match the number of rows in the other matrix.
        """
        assert self.block_cols == other.block_rows, "Block matrix dimensions must match for multiplication"
        result_blocks = [[np.zeros(self.block_size) for _ in range(other.block_cols)] for _ in range(self.block_rows)]
        for i in range(self.block_rows):
            for j in range(other.block_cols):
                for k in range(self.block_cols):
                    result_blocks[i][j] += self.diagonal_blocks[i][k] * other.diagonal_blocks[k][j]
        return DiagonalBlockMatrix(result_blocks)

    def is_block_diagonal(self):
        """
        Checks if the matrix is block diagonal.

        Returns:
            bool: True if the matrix is block diagonal, False otherwise.
        """
        for i in range(self.block_rows):
            for j in range(self.block_cols):
                if i != j and np.any(self.diagonal_blocks[i][j]):
                    return False
        return True

    def invert(self):
        """
        Inverts the current block matrix. If the matrix is block diagonal, it uses a more efficient method.
        Otherwise, it converts to dense form and inverts using numpy.

        Returns:
            DiagonalBlockMatrix or np.array: The inverted block matrix or its dense representation.
            
        Raises:
            ValueError: If any diagonal block contains zero elements (non-invertible).
            np.linalg.LinAlgError: If the dense matrix is not invertible.
        """
        if self.is_block_diagonal():
            inverted_blocks = []
            for i in range(self.block_rows):
                inverted_row = []
                for j in range(self.block_cols):
                    if i == j:
                        block = self.diagonal_blocks[i][j]
                        if np.any(block == 0):
                            raise ValueError("Matrix contains zero elements in a diagonal block, cannot be inverted")
                        inverted_block = 1.0 / block
                        inverted_row.append(inverted_block)
                    else:
                        inverted_row.append(np.zeros(self.block_size))
                inverted_blocks.append(inverted_row)
            return DiagonalBlockMatrix(inverted_blocks)
        else:
            dense_matrix = self.to_dense()
            try:
                inverted_dense_matrix = np.linalg.inv(dense_matrix)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("The matrix is not invertible.")
            return inverted_dense_matrix
