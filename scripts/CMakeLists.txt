add_executable(generate_aot_kernels generate_aot_kernels.cpp)
target_link_libraries(generate_aot_kernels PRIVATE dmk_bootstrap)
# generate_aot_kernels is a pure C++ code generator with no CUDA device code.
# Disable CUDA device linking so CMake does not generate cmake_device_link.o
# (which requires /lib64/libpthread.so, absent on glibc >= 2.34).
set_target_properties(generate_aot_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

add_executable(benchmark_gradient benchmark_gradient.cpp)
target_link_libraries(benchmark_gradient PRIVATE dmk_shared)
target_compile_definitions(benchmark_gradient PRIVATE DOCTEST_CONFIG_DISABLE SCTL_PROFILE=-1)
