# HPC
High Performance Computing

In this project we look at different aspects of high performance computing.

### :white_check_mark: __OpenMP Parallelisation__

#### 1. Parallelising KNN's distance and sorting functions.
  - Distance functions: Manhattand and Euclidean distances are used to calculate the k-nearest neighbours of a given point.
  - Sorting algorihtms: Quick sort and merge sort are the main sorting algorithms used to sort the calculated distance matrix to identify nearest neighbours for each given query point. Bubble sort, insertion sort and selection sort are also included to assess their performance in parallel version of KNN, if it is possible.
 
### :white_check_mark: __CUDA Parallelisation__
####  1. Matrix Transpose : Matrix transpose optimised using CUDA. 
The transpose of a matrix is an operator which flips a matrix over its diagonal. The row and column indices of the matrix is switched to produce a new matrix. The following versions of Matrix Transpose are implemented:
- Sequential/Serial verion (used as the base line to assess parallel implementations)
- Global memory version (parallelised using the global memory)
- Shared memory version (parallelised using the shared memory)

#### 2. Vector Reduction : Vector reduction (addition) optimised using CUDA. 
The vector reduction is a type of operator that is used to reduce the elements of a vectore to a single value by summing all the values. The following versions of Vector Reduction are implemented:
- Sequential/Serial verion (used as the base line to assess parallel implementations)
- Global memory version (parallelised using the global memory)
- Shared memory version (parallelised using the shared memory)

#### 3. Matrix Multiplication : Matrix multiplication optimised using CUDA. 
The matrix multiplication is a binary operation that produces a matrix from two matrices which is known as matrix product. The following versions of Matrix Multiplication are implemented:
- Sequential/Serial verion (used as the base line to assess parallel implementations)
- Global memory version (parallelised using the global memory)
- Shared memory version (parallelised using the shared memory)

