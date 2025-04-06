from typing import Any, Dict, List

import tensorflow as tf

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices("GPU")
print(f"Physical devices: {physical_devices}")
if physical_devices:
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set to True for GPU: {gpu}")
        except Exception as e:
            print(f"Error setting memory growth: {e}")

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"List of available GPUs: {tf.config.list_physical_devices('GPU')}")

# More detailed device information
print("\nDevice details:")
print(tf.config.list_physical_devices())

# Display device placement for a simple operation
print("\nDevice placement for a simple operation:")
strategy = (
    tf.distribute.MirroredStrategy()
    if physical_devices
    else tf.distribute.get_strategy()
)
print(f"Strategy: {strategy.__class__.__name__}")

with strategy.scope():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print(c)
    print(f"Device used: {c.device}")

# Print TensorFlow build info
print("\nTensorFlow build info:")
build_info = tf.sysconfig.get_build_info()
print(f"CUDA version: {build_info['cuda_version']}")
print(f"cuDNN version: {build_info['cudnn_version']}")
if "nccl_version" in build_info:
    print(f"NCCL version: {build_info['nccl_version']}")
else:
    print("NCCL version: Not available")
print(f"Is CUDA available: {tf.test.is_built_with_cuda()}")

# Run a simple test to check GPU performance
print("\nRunning a simple test on available device:")
import time

import numpy as np

# Create large matrices
matrix_size = 5000
A = tf.random.normal((matrix_size, matrix_size))
B = tf.random.normal((matrix_size, matrix_size))

# Time matrix multiplication
start_time = time.time()
C = tf.matmul(A, B)
_ = C.numpy()  # Force execution
end_time = time.time()

print(f"Matrix multiplication took: {end_time - start_time:.4f} seconds")
