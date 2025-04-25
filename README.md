# TurboBlur
Fast Gaussian blur using OpenMP threads and AVX2 SIMD for parallel image processing.


A high-performance image processing system implementing Gaussian blur using OpenMP and AVX for dramatic speedups over sequential processing.

This project demonstrates the power of parallel computing by optimizing a standard Gaussian blur algorithm using thread-level parallelism and SIMD instructions, achieving up to 55.9x speedup over traditional sequential implementation.

---

## 📁 Project Structure

```
.
├── sequential_blur.cpp                # Sequential implementation
├── parallel_blur.cpp                  # Parallel implementation (OpenMP + AVX)
├── plot_results.py                    # Performance visualization script
├── README.md                          # Build configuration
├── Images/                            # Input image directory
├── blurred_images_sequential/         # Output directory for sequential results
├── blurred_images_avx_openmp/         # Output directory for parallel results
├── sequential_times.csv               # Performance data (sequential)
└── parallel_times.csv                 # Performance data (parallel)
```

---

## 🧠 Objectives

- Implement Gaussian blur using both sequential and parallel approaches
- Leverage OpenMP for multi-threading parallelism
- Utilize AVX (Advanced Vector Extensions) for SIMD operations
- Compare performance across varying dataset sizes
- Demonstrate scalability of parallel implementation

---

## 📦 Dataset Information

The project processes image datasets of varying sizes:
- 125, 250, 500, 1000, 2000, 4000, and 5000 images
- Images should be named sequentially as `000000.jpg`, `000001.jpg`, etc.
- Based on a Kaggle dataset ("Label Me 12 50k")

---

## 🧰 Tools & Libraries Used

- C++17
- OpenCV
- OpenMP
- AVX (Advanced Vector Extensions)
- CMake
- Python with matplotlib and numpy

---

## ⚙️ Implementation Details

### Gaussian Kernel Generation

Both implementations use the same function to generate a Gaussian kernel:

```cpp
vector<vector<float>> generateGaussianKernel(int size, float sigma) {
    vector<vector<float>> kernel(size, vector<float>(size));
    float sum = 0.0;
    int half = size / 2;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            value /= (2 * CV_PI * sigma * sigma);
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    return kernel;
}
```

### Key Optimizations

The parallel implementation uses:
- `#pragma omp parallel for` to parallelize loops
- `__m256` AVX registers for processing multiple pixels simultaneously
- AVX intrinsics like `_mm256_add_ps`, `_mm256_mul_ps` for SIMD operations

---

## 🔧 How to Build

### Using CMake (recommended)

1. Create a CMakeLists.txt file:

```cmake
cmake_minimum_required(VERSION 3.10)
project(GaussianBlurParallel)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mavx")

add_executable(sequential_blur sequential_blur.cpp)
target_link_libraries(sequential_blur ${OpenCV_LIBS})

add_executable(parallel_blur parallel_blur.cpp)
target_link_libraries(parallel_blur ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
```

2. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

### Manual Compilation

Compile the sequential version:
```bash
g++ -O3 -std=c++17 sequential_blur.cpp -o sequential_blur `pkg-config --cflags --libs opencv4`
```

Compile the parallel version:
```bash
g++ -O3 -std=c++17 -fopenmp -mavx parallel_blur.cpp -o parallel_blur `pkg-config --cflags --libs opencv4`
```

---

## ⚠️ Important Code Fix

In the sequential_blur.cpp file, there's an issue with the image saving condition:

```cpp
// Change this line:
if (numImages == 160) {
    
// To this:
if (numImages == 1000) {  // Or any value that exists in datasetSizes
```

This fixes the issue where blurred images weren't being saved because 160 isn't in the datasetSizes array.

---

## 📈 Performance Results

| Number of Images | Sequential Time (s) | Parallel Time (s) | Speedup |
|------------------|--------------------:|------------------:|--------:|
| 125              | 3.36082             | 0.163751          | 20.5x   |
| 250              | 6.82529             | 0.17564           | 38.9x   |
| 500              | 15.2146             | 0.333934          | 45.6x   |
| 1000             | 35.3491             | 0.759924          | 46.5x   |
| 2000             | 70.9737             | 1.27034           | 55.9x   |
| 4000             | 124.384             | 2.4706            | 50.3x   |
| 5000             | 160.727             | 4.48554           | 35.8x   |

The parallel implementation achieves a maximum speedup of approximately 55.9x for 2000 images, demonstrating the effectiveness of combining OpenMP thread-level parallelism with AVX SIMD instructions.

---

## 🚀 Future Work

- Implement cache-aware tiling to improve memory access patterns
- Experiment with different OpenMP scheduling strategies
- Explore GPU implementation using CUDA or OpenCL
- Add support for different kernel sizes and sigma values
- Create a GUI interface for easier parameter tuning

---

## ✍️ Authors

- [Abhilash Surapuchetty]

---

## 📄 License

[MIT License]

## 📬 Contact

Feel free to reach out for collaboration, queries, or feedback!

- **📧 Email**: [alash0849@gmail.com](mailto:alash0849@gmail.com)
- **🔗 LinkedIn**: [Abhilash Surapuchetty](https://www.linkedin.com/in/abhilash-surapuchetty-baa0a4267/)