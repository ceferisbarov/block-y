## Problem statement
See: [Problem statement](https://hal.cse.msu.edu/misc/join/)

## Approach
Some observations:
* Each block is diagonal on its own.
* The whole matrix is **not necessarily** diagonal, i.e. it is not a block diagonal matrix.
* All blocks have the same dimensions.

Since each block is diagonal, we can store it as a vector. This reduces space requirements from $n^2$ to $n$. Let's see how we can perform addition and multiplication with this data structure.

## Addition
Addition is the simplest operation for matrices. It is performed by adding matrices element-wise. Our data structure allows this. In order to add matrices $A$ and $B$:  
  
$C = A + B$

We have to add blocks element-wise. An each of these additions consist of adding two diagonal vectors $V^A_{ij}$ and $V^B_{ij}$ element-wise:

$C[i][j] = V^A_{ij} + V^B_{ij}$
## Multiplication

If we are multiplying matrices $A$ and $B$:

$C = A × B$

Then each element of $C$ is calculated as follows:

$C[i][j]=\Sigma^{n-1}_{k=0} A[i][k]×B[k][j]$

Considering that both $A[i][k]$ and $B[i][k]$ are diagonal matrices. We can calculate $A[i][k]×B[k][j]$ by taking element-wise multiplication of their diagonals. This allows us to calculate $C$ by only using the diagonal values as vectors.

## Inversion
I have not come up with a general method for inversion. I implemented this function for the sake of completeting the work. It works as follows:
* If our matrix is block diagonal (which is not necessarily the case), we take reciprocal of each diagonal vector: $V_{ij} = 1 / V_{ij}$ where $i = j$.
* Otherwise, we get the dense version of the matrix and use NumPy for inversion.

## Implementation
The data structure has been implemented in pure Python. Numpy has been used only for utility functions such as `fill_diagonal` (except for the inversion method).

We have two test suites. The first tests against expected results typed in by me personally. The second tests against NumPy results.
