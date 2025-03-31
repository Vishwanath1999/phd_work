import numpy as np
from numba import cuda, float32

@cuda.jit
def dtw_cuda(x, y, dtw_matrix):
    i, j = cuda.grid(2)  # 2D grid of threads
    
    if i == 0 or j == 0:  # Skip the padding row and column
        return

    n, m = dtw_matrix.shape
    if i < n and j < m:
        cost = abs(x[i - 1] - y[j - 1])
        dtw_matrix[i, j] = cost + min(
            dtw_matrix[i - 1, j],    # Insertion
            dtw_matrix[i, j - 1],    # Deletion
            dtw_matrix[i - 1, j - 1] # Match
        )

# Initialize data
x = np.random.rand(3000).astype(np.float32)
y = np.random.rand(3000).astype(np.float32)
n, m = len(x) + 1, len(y) + 1
dtw_matrix = np.full((n, m), 1e9, dtype=np.float32)  # Use large finite value
dtw_matrix[0, 0] = 0.0  # Start point

# Allocate GPU memory
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_dtw_matrix = cuda.to_device(dtw_matrix)

# Define thread/block layout
threads_per_block = (16, 16)
blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (m + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Run kernel
dtw_cuda[blocks_per_grid, threads_per_block](d_x, d_y, d_dtw_matrix)

# Copy result back
dtw_matrix = d_dtw_matrix.copy_to_host()

# Final DTW distance
dtw_distance = dtw_matrix[-1, -1]
print("DTW Distance:", dtw_distance)
