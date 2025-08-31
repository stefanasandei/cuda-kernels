# CUDA Kernels

A collection of high-performance CUDA kernels for parallel computing tasks. I'm writing these as I learn CUDA.

## Kernels

| kernel     | time             | reference impl.  |
|------------|------------------|------------------|
| vector add | 0.087 ± 0.039 ms | 0.052 ± 0.001 ms |

Measured on my RTX 2060 discrete GPU. Size of arrays found in tests (`./src/tests`), also same sizes between my kernels
and the PyTorch/Triton ones.

## Building the Project

Prerequisites:

- **CUDA Toolkit**: Version 12.0 or later
- **CMake**: Version 4.0 or later
- **C++ Compiler**: GCC 11+ or Clang 14+ with C++23 support
- **NVIDIA GPU**: With compute capability 7.5 or higher

1\. **Clone the repository**:

   ```bash
   git clone https://github.com/stefanasandei/cuda-kernels.git --recursive
   cd cuda-kernels
   ```

If you forgot the `--recursive` flag, run this to fetch submodules:

   ```bash
   git submodule update --init --recursive
   ```

2\. **Create build directory**:

   ```bash
   mkdir build && cd build
   ```

3\. **Configure with CMake**:

   ```bash
   cmake ..
   ```

4\. **Build the project**:

   ```bash
   make -j$(nproc)
   ```

Afterwards you can run the tests:

```bash
./tests/tests
```

## Project Structure

```
cuda-kernels/
├── CMakeLists.txt          # Root build configuration
├── lib/
│   ├── CMakeLists.txt      # Library build config
│   ├── common/             # Common utility files
│   └── googletest/         # Google Test framework
├── src/
│   ├── CMakeLists.txt      # Source build config
│   └── example_kernel/
│       └── example_kernel.cu   # kernel implementation
└── tests/
    ├── CMakeLists.txt      # Test build configuration
    └── example_kernel.cpp   # tests for the kernel
```

### Adding New Kernels

1. Create a new directory in `src/` for your kernel
2. Implement the kernel in a `.cu` file
3. Add the host wrapper function declaration in the `./src/kernels.h` file
4. Add unit tests in `tests/`, a cpp file that calls the host wrapper

The `vector_add` kernel is the simplest one, as an example for the implementation.

## License

[MIT](LICENSE) © [Asandei Stefan-Alexandru](https://asandei.com). All rights reserved.
