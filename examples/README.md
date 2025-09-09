# Files submitted by Giorgos Kementzidis.
Date: August 7, 2025.

### Modules (on FI resources):
- modules/2.4-beta2 
- python 
- gcc/13 
- openmpi 
- intel-oneapi-mkl 
- intel-oneapi-compilers

### Build instructions
```
git checkout ewald
git submodule init
git submodule update

module load python gcc openmpi intel-api-mkl intel-oneapi-compilers
mkdir build
cd build

# you may need to run this twice if it fails. there's an nda cmake bug...
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBLA_VENDOR=Intel10_64lp_seq

# this builds only the ewald_final_test object
make ewald_final_test -j10
```

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
