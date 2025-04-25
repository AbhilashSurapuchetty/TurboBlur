import matplotlib.pyplot as plt
import numpy as np

# Sequential Results (Data from your terminal)
sequential_images = np.array([125, 250, 500, 1000, 2000, 4000, 5000])
sequential_times = np.array([3.36082, 6.82529, 15.2146, 35.3491, 70.9737, 124.384, 160.727])
sequential_throughput = sequential_images / sequential_times  # Throughput = Images / Time

# Parallel Results (Data from your terminal)
parallel_images = np.array([125, 250, 500, 1000, 2000, 4000, 5000])
parallel_times = np.array([0.163751, 0.17564, 0.333934, 0.759924, 1.27034, 2.4706, 4.48554])
parallel_throughput = parallel_images / parallel_times  # Throughput = Images / Time

# Calculate Speedup (Sequential Time / Parallel Time)
speedup = sequential_times / parallel_times

# Plot Throughput Comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sequential_images, sequential_throughput, label='Sequential', marker='o', color='r')
plt.plot(parallel_images, parallel_throughput, label='Parallel', marker='x', color='g')
plt.title('Throughput vs Number of Images')
plt.xlabel('Number of Images')
plt.ylabel('Throughput (images/second)')
plt.legend()

# Plot Speedup vs Number of Images
plt.subplot(1, 2, 2)
plt.plot(sequential_images, speedup, label='Speedup', marker='o', color='b')
plt.title('Speedup vs Number of Images')
plt.xlabel('Number of Images')
plt.ylabel('Speedup (Sequential/Parallel)')
plt.legend()

plt.tight_layout()
plt.show()
