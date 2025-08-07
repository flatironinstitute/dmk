# Files submitted by Giorgos Kementzidis.
Date: August 7, 2025.

### Modules (on FI resources):
- modules/2.4-beta2 
- python 
- gcc/13 
- openmpi 
- intel-oneapi-mkl 
- intel-oneapi-compilers

### The final version of the different files I created are the following:

- `ewald_total.cpp`
- `ewald_total.h`
- `test_ewald.c`
- `fft_parallel_2D_factored.cpp`
- `fft_parallel_3D.cpp`
- `parallel_transpose.cpp`
- `parallel_transpose_3D.cpp`

## Some "legacy" code:
Initial scripts with possibly slower and less organized implementations. At the top of the documents you can read descriptions about what they contain.

- `ewald_test.cpp`
- `ewald_short.cpp`
- `ewald_shorter.cpp`
- `fft_parallel.cpp`
- `fft_parallel_v.cpp`
- `fft_parallel_2D.cpp`

## Additional comments:

In `ewald_total.cpp`, you can choose to include either the `main()` function to execute the script, or the `extern "C"` part at the end for a C API implementation. After modifying the file and commenting/uncommenting the appropriate components, you can either run `make ewald_total` or `make ewald`, respectively. Running `make` when both `add_executable()` and `add_library()` are present in the same `CMakeLists.txt` file will probably lead to errors.

## Running C tests:
To run `test_ewald.c` make sure to have compiled the library code from `ewald_total.cpp` using `make ewald` and then run the following line in the directory where `test_ewald.c` is located:

`g++ test_ewald.c -L../build/examples -lewald -Wl,-rpath,'$ORIGIN/../build/examples' -o test_ewald`