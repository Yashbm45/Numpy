# NumPy stands for Numeric Python. 
# It is a powerful Python library used for numerical computing. It provides support for multi-dimensional arrays, along with a large collection of mathematical functions.

# Why NumPy --> fast, memory efficient comapared to Python
# Why fast --> Written in CPP, arrays are homogeneous and stored at continous memory location. Easy and efficient to access and manipulate.


# #### ✅ Advantages of NumPy
# Feature	                Description
# 1. Performance	        NumPy operations are written in C and optimized for speed. It's much faster than regular Python lists.
# 2. Memory Efficiency	Uses less memory than built-in lists (compact array data structure).
# 3. Array Broadcasting	Allows arithmetic operations on arrays of different shapes without looping.
# 4. Vectorization	    Eliminates the need for for loops via vectorized operations, making code cleaner and faster.
# 5. Multi-dimensional    Arrays	Supports n-dimensional arrays (not just 1D or 2D like lists).
# 6. Rich Functionality	Includes functions for linear algebra, Fourier transforms, random numbers, statistics, etc.
# 7. Integration	        Easily integrates with C/C++ and other scientific libraries like SciPy, Pandas, and TensorFlow.
# 8. Indexing & Slicing	Powerful tools for accessing and modifying data efficiently.



# #### ❌ Disadvantages of NumPy
# Issue	                    Description
# 1. Steeper Learning Curve	Beginners may find syntax and broadcasting complex at first.
# 2. Limited to Numeric Data	Not ideal for non-numeric data types like text or objects.
# 3. Fixed Size	            NumPy arrays are not as flexible as Python lists (size cannot be changed dynamically).
# 4. Less Flexibility with Heterogeneous Data	    --> All elements in a NumPy array must be of the same data type.
# 5. Requires Compilation	Some operations are lower-level and require understanding of C-style memory.




# #### 🛠️ Common Use Cases of NumPy


# 📊 1. Data Analysis & Data Science
# Efficient storage and computation of large datasets.
# Often used with Pandas for data manipulation.

# 🎲 2. Random Number Generation
# NumPy has its own random module for generating samples from different distributions.

# 🧮 3. Linear Algebra
# Solving equations, matrix multiplication, eigenvalues, etc.
# 📐 4. Scientific Computing
# Widely used in physics, chemistry, astronomy, etc.

# 🧠 5. Machine Learning / AI
# Libraries like TensorFlow and scikit-learn use NumPy internally.

# 🎮 6. Image Processing
# Images are treated as arrays (grayscale, RGB) and manipulated using NumPy.

# ⏱️ 7. Performance-Critical Computations
# Ideal for real-time systems where speed is critical.



# 🔍 NumPy vs Python List – Quick Comparison
# Feature     	NumPy Array	            Python List
# Performance 	Very fast (C-based)	    Slower (interpreted)
# Memory Usage	Lower	                Higher
# Type Safety	    Homogeneous types	    Heterogeneous types
# Operations	    Vectorized	            Manual loop required



# 🧪 Example: Vectorized Operation
#     Using Python list:
#         a = [1, 2, 3]
#         b = [4, 5, 6]
#             result = [x + y for x, y in zip(a, b)]
#     Using NumPy:
#         import numpy as np
#         a = np.array([1, 2, 3])
#         b = np.array([4, 5, 6])
#             result = a + b

# Array in NumPy
# Array is datatype in NumPy which stored homgenous values.
# Array can be one or multi-dimensional
# We can convert lists, tuples or any array like object into array
# import numpy as np

# # Creating a 1D array
# x = np.array([1, 2, 3])

# # Creating a 2D array
# y = np.array([[1, 2], [3, 4]])

# # Creating a 3D array
# z = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# print(x)
# print(y)
# print(z)
# ## 🧮 NumPy Array Creation Methods – Theory + Syntax
# 🔹 1. np.zeros(shape)   --> Creates an array filled with zeros.
#         Theory --> Used to initialize an array or matrix with all elements as 0. Useful for preallocating memory for computations.

#         Syntax: np.zeros(shape, dtype=float)



# 🔹 2. np.ones(shape)
#     Creates an array filled with ones.
#     Theory: Commonly used for initializing weights, bias matrices, or to test broadcasting.



# 🔹 3. np.eye(N)
#         Creates an identity matrix (square matrix with 1s on the diagonal and 0s elsewhere).
#         Useful in linear algebra for operations like finding inverses or solving systems of equations.
        
#         🔤 Syntax: np.eye(N, M=None, k=0, dtype=float)
#         N: number of rows
#         M: number of columns (defaults to N)
#         k: diagonal offset (0 = main diagonal)
        
#         Example: np.eye(3)
#         # Output: array([[1., 0., 0.],
#         #                [0., 1., 0.],
#         #                [0., 0., 1.]])



# 🔹 4. np.arange(start, stop, step)
#         Creates arrays with regularly spaced values, similar to range() in Python.
#         🧠 Theory: Efficient way to create sequences. Good for indexing or parameter sweeping.
#         🔤 Syntax: np.arange(start, stop, step, dtype=None)
#                 Step - is gap between two numbers.
#         🧪 Example: np.arange(0, 10, 2)
#         # Output: array([0, 2, 4, 6, 8])



# 🔹 5. np.linspace(start, stop, num)
#         ✅ Purpose: Creates evenly spaced values between two endpoints.
#         🧠 Theory: Ideal for plotting or mathematical modeling where you need a set number of points between a range.
#         🔤 Syntax: np.linspace(start, stop, num=50, endpoint=True, dtype=None)
#             start: starting value
#             stop: ending value
#             num: number of samples
#             endpoint=True: include stop value

#         🧪 Example: np.linspace(0, 1, 5)
#         # Output: array([0.  , 0.25, 0.5 , 0.75, 1.  ])



# 🧰 Bonus: Other Creation Functions
# Function	            Description
# np.full(shape, value)	Creates array filled with a constant value
# np.random.rand()	    Random samples from uniform distribution
# np.empty(shape)	        Uninitialized values (faster, for advanced use)
# np.copy(array)	        Copies an existing array
# # Array of Zeros
# zeros = np.zeros([4,3],int)
# print(zeros)

# # Array of Ones
# ones = np.ones([3,3],float)
# print(ones)

# # Identity Matrix
# I_matrix = np.eye(3)
# I_matrix2 = np.eye(3,3,1,int,'f')
# print(I_matrix)
# print(I_matrix2)

# # Arrange (start, stop, step(gap), datatype)
# arrange = np.arange(2,10,2,'float')
# print(arrange)

# # linspace 
# linespace = np.linspace(1,50,6,'false')
# print(linespace)
# ### Attributes of Array in NumPy
# When you create a NumPy array using np.array() or any other array-creation method, the resulting object is an instance of numpy.ndarray.


# 🔹 1. .ndim – Number of Dimensions - Returns the number of dimensions (axes) of the array.

# 🔹 2. .shape – Shape of the Array  - Returns a tuple indicating the size of each dimension.
        
#         🧪 Example: print(a.shape)  # Output: (2, 3) 🔸 Means 2 rows, 3 columns

# 🔹 3. .size – Total Number of Elements  -   Returns the total number of elements in the array.

# 🔹 4. .dtype – Data Type of Elements    -   Returns the data type of elements in the array.

# 🔹 5. .itemsize – Size of One Element in Bytes  - Returns the memory size (in bytes) of one array element.

# 🔹 6. .nbytes – Total Bytes Consumed    - Total memory used by the array = size * itemsize.

# 🔹 7. .T – Transpose of the Array   -   Returns a transposed version of the array (rows become columns and vice versa).

# 🔹 8. .data – Memory Buffer (Advanced)  - Returns the memory buffer containing the actual elements (rarely used directly).


# 🔍 Summary Table 


# * 🔹Attribute	        Description	Example Output
# * 🔹.ndim	        Number of dimensions	2
# * 🔹.shape	        Tuple of array dimensions	(2, 3)
# * 🔹.size	        Total number of elements	6
# * 🔹.dtype	        Data type of array elements	int64, float64
# * 🔹.itemsize	        Size in bytes of each element	8
# * 🔹.nbytes	        Total memory in bytes	48
# * 🔹.T	                Transpose of the array	Array with swapped axes
# * 🔹.data	        Memory buffer (rarely used)	<memory at 0x...>


# import numpy as np
# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a.ndim)  # Output: 2

# size = a.size
# print(f'Size is : {size}')

# print(f'Transpose of Array : {a.transpose()}')

# print(f'Shape of Array : {a.shape}')

# print(f'Data type of Array : {a.dtype}')
# # Converting Array Dimensions
# num = np.array([1,2,3,4,5], ndmin=3)
# print(num)
# # Reshaping Array
# re = np.array([1,2,3,4,5,6,7,8,9])

# print(re.reshape(3,3))
# print(re.reshape(3,3).base)     # View original Data

# di = np.array([1,2,3,4,5,6,7,8])
# print(di.reshape(2,2,-1))       # -1 is unknown dimension you can have only one unknown dimension

