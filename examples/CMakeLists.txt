if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example PRIVATE dmk)

  add_executable(measure_error measure_error.cpp)
  target_link_libraries(measure_error PRIVATE dmk)
endif()

if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP AND DMK_BUILD_PVFMM)
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm PRIVATE dmk pvfmmStatic)
endif()

add_executable(benchmark_kernels benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels PRIVATE dmk_shared)

if (DMK_BUILD_ESP)
  add_executable(benchmark_esp benchmark_esp.cpp)
  target_link_libraries(benchmark_esp PRIVATE dmk_shared)
  target_compile_definitions(benchmark_esp PRIVATE
    VERIFY_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/verify_esp.py"
    PERILAP3D_DIR="${CMAKE_SOURCE_DIR}/scripts/perilap3d"
    VERIFY_CACHE_DIR="${CMAKE_BINARY_DIR}"
  )

  add_executable(plot_esp_splitting plot_esp_splitting.cpp)
  target_link_libraries(plot_esp_splitting PRIVATE dmk_shared)

  add_executable(plot_esp_fourier plot_esp_fourier.cpp)
  target_link_libraries(plot_esp_fourier PRIVATE dmk_shared)

  # Standalone ducc0 FFT scaling test. Deliberately does NOT link dmk_shared --
  # no ESP code is involved, and the only library call it makes is ducc0::c2c.
  #
  # It links finufft::finufft purely to resolve a symbol: ducc0 explicitly
  # instantiates c2c<double> in its fft_inst*.cc, which the prebuilt libducc0.a
  # here does NOT contain (that archive has only threading/mav/string_utils) --
  # finufft's build is what compiles those units. Linking finufft is far cheaper
  # than recompiling ducc0's heavy fft_inst template units into this target, and
  # no FINUFFT code executes either way.
  #
  # Threads::Threads because ducc0's pool is std::thread-based; ducc0's own
  # CMakeLists-C++.txt template links it for the same reason (its
  # DUCC0_USE_THREADS option does exactly this and nothing else -- there is no
  # macro that switches ducc0's threading on or off at compile time).
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  add_executable(benchmark_ducc0_fft benchmark_ducc0_fft.cpp)
  target_link_libraries(benchmark_ducc0_fft PRIVATE ducc0 finufft::finufft Threads::Threads)
endif()

if (DMK_BUILD_ESP AND DMK_GPU_OFFLOAD)
  add_executable(benchmark_esp_gpu benchmark_esp_gpu.cpp)
  target_link_libraries(benchmark_esp_gpu PRIVATE dmk_shared CUDA::cudart)
endif()
