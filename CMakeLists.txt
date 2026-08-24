cmake_minimum_required(VERSION 3.28)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" DMK_VERSION)
string(STRIP "${DMK_VERSION}" DMK_VERSION)

# On glibc >= 2.34, libpthread is merged into libc. Force -pthread compiler flag
# rather than a full-path library, so CMake does not cache /lib64/libpthread.so.
set(THREADS_PREFER_PTHREAD_FLAG ON)

project(dmk VERSION ${DMK_VERSION} LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(DMK_GIT_COMMIT "unknown")
set(DMK_GIT_DESCRIBE "unknown")
find_package(Git QUIET)
if (Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE dmk_commit)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE dmk_describe)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE dmk_tag RESULT_VARIABLE dmk_tagged)
  if (dmk_commit)
    set(DMK_GIT_COMMIT "${dmk_commit}")
  endif()
  if (dmk_describe)
    set(DMK_GIT_DESCRIBE "${dmk_describe}")
  endif()
  if (dmk_tagged EQUAL 0 AND NOT dmk_tag STREQUAL "v${PROJECT_VERSION}")
    message(WARNING "git tag ${dmk_tag} disagrees with project version v${PROJECT_VERSION}")
  endif()
endif()

set(DMK_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
configure_file(include/dmk/version.h.in ${DMK_GENERATED_INCLUDE_DIR}/dmk/version.h @ONLY)

option(DMK_BUILD_TESTS "Build tests" OFF)
option(DMK_BUILD_EXAMPLES "Build examples" ON)
option(DMK_INSTRUMENT "Instrument the code" OFF)
option(DMK_BUILD_PVFMM "Build PVFMM for comparison. Useless without DMK_BUILD_EXAMPLES" OFF)
option(DMK_HAVE_MPI "Use MPI for parallelism" ON)

option(DMK_HAVE_OPENMP "Use OpenMP for parallelism" ON)
option(DMK_BUILD_REFERENCE "Build reference Fortran implementation" OFF)
option(DMK_USE_JIT "Use JIT compilation where supported (requires LLVM)" OFF)
option(DMK_GPU_OFFLOAD "Enable GPU offloading via CUDA" OFF)

if (DMK_BUILD_REFERENCE)
  enable_language(Fortran)
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(DMK_VECLIB_DEFINE "SCTL_HAVE_SVML")
else()
  set(DMK_VECLIB_DEFINE "SCTL_HAVE_LIBMVEC")
endif()

# Disable google tests and HDF5 for NDA
set(Build_Tests OFF CACHE BOOL "" FORCE)
set(HDF5Support OFF CACHE BOOL "" FORCE)

set(DMK_CXX_FLAGS "-DSCTL_MAX_DEPTH=62;-D${DMK_VECLIB_DEFINE}")
if (DMK_BUILD_TESTS)
  set(DMK_TEST_SRC src/doctest.cpp)
else()
  set(DMK_TEST_SRC)
  list(APPEND DMK_CXX_FLAGS "-DDOCTEST_CONFIG_DISABLE")
endif()

if (DMK_INSTRUMENT)
  list(APPEND DMK_CXX_FLAGS "-DSCTL_PROFILE=5" "-DDMK_INSTRUMENT")
endif()

set(DMK_OPENMP_TARGET "")
if (DMK_HAVE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(DMK_OPENMP_TARGET OpenMP::OpenMP_CXX)
  list(APPEND DMK_CXX_FLAGS "-DDMK_HAVE_OPENMP")
  set(OpenMPSupport ON CACHE BOOL "" FORCE) # extern/nda's own option name
  set(FINUFFT_USE_OPENMP ON CACHE BOOL "" FORCE)
else()
  set(OpenMPSupport OFF CACHE BOOL "" FORCE)
  set(FINUFFT_USE_OPENMP OFF CACHE BOOL "" FORCE)
endif()

set(DMK_MPI_TARGET "")
if (DMK_HAVE_MPI)
  find_package(MPI REQUIRED)
  set(DMK_MPI_TARGET MPI::MPI_CXX)
  list(APPEND DMK_CXX_FLAGS "-DSCTL_HAVE_MPI" "-DDMK_HAVE_MPI")
  set(MPISupport ON CACHE BOOL "" FORCE)
else()
  set(MPISupport OFF CACHE BOOL "" FORCE)
endif()

set(DMK_GPU_TARGET "")
if (DMK_GPU_OFFLOAD)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(CUDAToolkit REQUIRED)
  set(DMK_GPU_TARGET CUDA::cudart)
  set(DMK_CUDA_JIT_TARGETS CUDA::cuda_driver CUDA::nvrtc)
  set(DMK_CUDA_JIT_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  # Root that embedded JIT source keys are relative to; feeds both the embed script and
  # DMK_JIT_SOURCE_DIR, which have to agree.
  set(DMK_JIT_EMBED_ROOT "${PROJECT_SOURCE_DIR}/src/cuda")
  configure_file(
    ${PROJECT_SOURCE_DIR}/src/cuda/jit/dmk_jit_config.hpp.in
    ${DMK_CUDA_JIT_GENERATED_INCLUDE_DIR}/dmk_jit_config.hpp
    @ONLY
  )

  file(GLOB DMK_JIT_CU_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/cuda/*/jit_sources/*.cu")
  file(GLOB DMK_JIT_HEADER_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/dmk/cuda/*_kernelargs.hpp")
  # Device-only headers a JIT source may include that are not kernel-arg structs.
  list(APPEND DMK_JIT_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/dmk/cuda/bessel_device.hpp")
  set(DMK_JIT_EMBED_HEADERS "")
  foreach(hfile ${DMK_JIT_HEADER_FILES})
    get_filename_component(hname "${hfile}" NAME)
    list(APPEND DMK_JIT_EMBED_HEADERS "dmk/cuda/${hname}=${hfile}")
  endforeach()
  set(DMK_JIT_EMBEDDED_SOURCES "${DMK_CUDA_JIT_GENERATED_INCLUDE_DIR}/dmk_jit_embedded_sources.cpp")

  string(REPLACE ";" "|" DMK_JIT_CU_SOURCES_ARG "${DMK_JIT_CU_SOURCES}")
  string(REPLACE ";" "|" DMK_JIT_EMBED_HEADERS_ARG "${DMK_JIT_EMBED_HEADERS}")

  add_custom_command(
    OUTPUT ${DMK_JIT_EMBEDDED_SOURCES}
    COMMAND ${CMAKE_COMMAND}
      -DEMBED_OUTPUT=${DMK_JIT_EMBEDDED_SOURCES}
      -DEMBED_CU=${DMK_JIT_CU_SOURCES_ARG}
      -DEMBED_HEADERS=${DMK_JIT_EMBED_HEADERS_ARG}
      -DEMBED_ROOT=${DMK_JIT_EMBED_ROOT}
      -P ${PROJECT_SOURCE_DIR}/cmake/embed_jit_sources.cmake
    DEPENDS ${DMK_JIT_CU_SOURCES} ${DMK_JIT_HEADER_FILES}
      ${PROJECT_SOURCE_DIR}/cmake/embed_jit_sources.cmake
    VERBATIM
  )

  list(APPEND DMK_CXX_FLAGS "-DDMK_GPU_OFFLOAD")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

set(DMK_CUDA_JIT_SRC
  ${DMK_JIT_EMBEDDED_SOURCES}
  src/cuda/jit/autotune.cpp
  src/cuda/jit/jit_cache.cpp
  src/cuda/jit/jit_compiler.cpp
  src/cuda/jit/jit_kernel.cpp
  src/cuda/jit/jit_source_utils.cpp
)

find_package(BLAS REQUIRED)

if (DMK_BUILD_PVFMM)
  include(FetchContent)
  FetchContent_Declare(
    pvfmm_lib
    GIT_REPOSITORY https://github.com/dmalhotra/pvfmm.git
    GIT_TAG develop
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(pvfmm_lib)
endif()

set(DMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
  ${DMK_GENERATED_INCLUDE_DIR}
)

include(FetchContent)
# Vendored dependencies are archived into libdmk.a and linked into libdmk.so, so they must be built
# static and PIC. Normal variables, so a cached BUILD_SHARED_LIBS cannot override them.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FINUFFT_USE_DUCC0 ON CACHE BOOL "" FORCE)
set(DMK_GPU_ESP_TARGET "")
set(DMK_GPU_ESP_INTERFACE "")
if (DMK_GPU_OFFLOAD)
  set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)
  # Kept off dmk_internal: that feeds generate_aot_kernels, which must not pull cufinufft.
  set(DMK_GPU_ESP_TARGET finufft::cufinufft CUDA::cufft)
  # What a consumer of libdmk.a must name itself; cufinufft is archived in (see DMK_BUNDLE_TARGETS).
  set(DMK_GPU_ESP_INTERFACE CUDA::cufft)
endif()
FetchContent_Declare(
  finufft
  GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
  GIT_TAG v2.5.1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(finufft)
if (DMK_GPU_OFFLOAD)
  # finufft builds cufinufft with CUDA_SEPARABLE_COMPILATION, leaving undefined
  # __cudaRegisterLinkedBinary_* stubs that only an nvlink step supplies. libdmk links as CXX and
  # libdmk.a is a plain archive, so neither would run it -- resolve them into cufinufft's own archive.
  set_property(TARGET cufinufft PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
set(DMK_ESP_TARGET finufft::finufft ducc0)
set(DMK_ESP_SRC src/esp.cpp)

FetchContent_Declare(
  polyfit
  GIT_REPOSITORY https://github.com/DiamonDinoia/polyfit.git
  GIT_TAG        a2994771155ed4cd22c2b792b463c42eefca3067
  EXCLUDE_FROM_ALL
)
set(MONOFIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MONOFIT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(polyfit)

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/doctest EXCLUDE_FROM_ALL)
add_subdirectory(extern/nda EXCLUDE_FROM_ALL)
set(DMK_JIT_TARGET "")
if (DMK_USE_JIT)
  list(APPEND DMK_CXX_FLAGS "-DDMK_USE_JIT")
  set(RUFUS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(RUFUS_LINK_LLVM_SHARED ON CACHE BOOL "" FORCE)
  # librufus.a is bundled into libdmk.a, so its libLLVM.so dependency becomes ours. RuFuS's own
  # find_package defines the imported target in its directory scope only.
  find_package(LLVM REQUIRED CONFIG)
  set(DMK_JIT_TARGET LLVM)
  add_subdirectory(extern/RuFuS EXCLUDE_FROM_ALL)

  rufus_embed_ir_as_header(jit_kernels src/jit_kernels.cpp
    INCLUDES ${DMK_INCLUDES}
    DEFINITIONS ${DMK_VECLIB_DEFINE}
    FROM_TARGET nda::nda_c
  )
endif()

if (DMK_BUILD_REFERENCE)
  list(APPEND DMK_CXX_FLAGS "-DDMK_HAVE_REFERENCE")
  file(GLOB_RECURSE DMK_REF_COMMON_SRC CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.f*"
    "${PROJECT_SOURCE_DIR}/vec-kernels/src/libkernels2.cpp")
  add_library(dmk_ref_common OBJECT ${DMK_REF_COMMON_SRC})
  set_target_properties(dmk_ref_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(dmk_ref_common PRIVATE
    "${PROJECT_SOURCE_DIR}/extern/VCL/version2"
    "${PROJECT_SOURCE_DIR}/vec-kernels/include")

  file(GLOB_RECURSE DMK_REF_PDMK_SRC CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/pdmk/*.f")
  list(REMOVE_ITEM DMK_REF_PDMK_SRC
    "${PROJECT_SOURCE_DIR}/src/pdmk/kernelevaluation/l3dkernels_fast.f"
    "${PROJECT_SOURCE_DIR}/src/pdmk/pdmk.f"
    "${PROJECT_SOURCE_DIR}/src/pdmk/pdmk3.f"
  )
  add_library(dmk_ref_pdmk OBJECT ${DMK_REF_PDMK_SRC})
  set_target_properties(dmk_ref_pdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  file(GLOB_RECURSE DMK_REF_BDMK_SRC CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/bdmk/*.f")
  list(REMOVE_ITEM DMK_REF_BDMK_SRC
    "${PROJECT_SOURCE_DIR}/src/bdmk/bdmk4.f"
  )
  add_library(dmk_ref_bdmk OBJECT ${DMK_REF_BDMK_SRC})
  set_target_properties(dmk_ref_bdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(dmk_ref STATIC
    $<TARGET_OBJECTS:dmk_ref_common>
    $<TARGET_OBJECTS:dmk_ref_pdmk>
    $<TARGET_OBJECTS:dmk_ref_bdmk>)
endif()

set(DMK_BOOTSTRAP_SRC src/dmk.cpp src/proxy.cpp src/tree.cpp src/logger.cpp src/util.cpp src/legeexps.cpp
  src/fourier_data.cpp src/tensorprod.cpp src/planewave.cpp src/direct.cpp src/prolate.cpp src/prolate0_eval.cpp
  src/timing.cpp
  ${DMK_TEST_SRC}
  ${DMK_ESP_SRC}
)
# Generated by scripts/generate_aot_kernels, one translation unit per getter
file(GLOB DMK_AOT_KERNEL_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/aot/aot_kernels_*.cpp)
if (NOT DMK_AOT_KERNEL_SRC)
  message(FATAL_ERROR "No generated AOT kernels in src/aot. Restore them with\n"
                      "  git restore src/aot\n"
                      "or, from a build tree that already configures, regenerate from the project root:\n"
                      "  ./build/scripts/generate_aot_kernels")
endif()
set(DMK_AOT_SRC src/aot_evaluator.cpp ${DMK_AOT_KERNEL_SRC})

# Everything needed to compile against the C++ headers, but not to install
add_library(dmk_internal INTERFACE)
target_include_directories(dmk_internal INTERFACE ${DMK_INCLUDES})
target_compile_options(dmk_internal INTERFACE "${DMK_CXX_FLAGS}")
target_link_libraries(dmk_internal INTERFACE ${DMK_OPENMP_TARGET} ${DMK_MPI_TARGET} ${DMK_GPU_TARGET}
  ${DMK_CUDA_JIT_TARGETS} ${DMK_ESP_TARGET}
  nda::nda_c spdlog::spdlog_header_only doctest::doctest BLAS::BLAS polyfit::polyfit
)
if (DMK_USE_JIT)
  target_link_libraries(dmk_internal INTERFACE RuFuS::rufus jit_kernels)
endif()

add_library(DMKOBJS_BOOTSTRAP OBJECT ${DMK_BOOTSTRAP_SRC})
set_target_properties(DMKOBJS_BOOTSTRAP PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(DMKOBJS_BOOTSTRAP PUBLIC dmk_internal)

add_library(DMKOBJS_AOT OBJECT ${DMK_AOT_SRC})
set_target_properties(DMKOBJS_AOT PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(DMKOBJS_AOT PUBLIC DMKOBJS_BOOTSTRAP)

add_library(dmk_bootstrap STATIC)
target_link_libraries(dmk_bootstrap PUBLIC DMKOBJS_BOOTSTRAP)

if (DMK_GPU_OFFLOAD)
  add_library(DMKOBJS_CUDA OBJECT
              src/cuda/esp/esp_gpu.cpp
              src/cuda/esp/short_range.cpp
              src/cuda/esp/support.cpp
              src/cuda/esp/sort.cu
              src/cuda/pt/state.cpp
              src/cuda/pt/tree.cpp
              src/cuda/pt/direct.cpp
              src/cuda/pt/upward.cpp
              src/cuda/pt/tensorprod.cpp
              src/cuda/pt/form_outgoing.cpp
              src/cuda/pt/downward.cpp
              src/cuda/pt/eval_targets.cpp
              src/cuda/pt/self_correction.cpp
              src/cuda/pt/proxy2pw.cpp
              src/cuda/pt/pw2proxy.cpp
              src/cuda/pt/launchers.cpp
              src/cuda/direct/direct_freespace.cpp
              ${DMK_CUDA_JIT_SRC}
              )
  set_target_properties(DMKOBJS_CUDA PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(DMKOBJS_CUDA PUBLIC
    ${DMK_INCLUDES}
  )

  target_include_directories(DMKOBJS_CUDA PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${DMK_CUDA_JIT_GENERATED_INCLUDE_DIR}
  )

  target_link_libraries(DMKOBJS_CUDA PUBLIC
    ${DMK_GPU_TARGET}
    ${DMK_GPU_ESP_TARGET}
    ${DMK_CUDA_JIT_TARGETS}
    ${DMK_OPENMP_TARGET}
    ${DMK_MPI_TARGET}
    nda::nda_c
    spdlog::spdlog_header_only
  )
  target_compile_options(DMKOBJS_CUDA PUBLIC "${DMK_CXX_FLAGS}")
endif()

set(DMK_OBJECTS $<TARGET_OBJECTS:DMKOBJS_BOOTSTRAP> $<TARGET_OBJECTS:DMKOBJS_AOT>)
if (DMK_GPU_OFFLOAD)
  list(APPEND DMK_OBJECTS $<TARGET_OBJECTS:DMKOBJS_CUDA>)
endif()

# Sources rather than linked object libraries: an exported target may not name a non-exported one.
add_library(dmk STATIC ${DMK_OBJECTS})
add_library(dmk_shared SHARED ${DMK_OBJECTS})
target_link_libraries(dmk_shared PRIVATE dmk_internal ${DMK_GPU_ESP_TARGET})

foreach(dmk_target dmk dmk_shared)
  set_target_properties(${dmk_target} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${dmk_target} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${DMK_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  if (DMK_HAVE_MPI)
    target_compile_definitions(${dmk_target} INTERFACE DMK_HAVE_MPI)
  endif()
endforeach()

set_target_properties(dmk PROPERTIES EXPORT_NAME dmk_static)
set_target_properties(dmk_shared PROPERTIES OUTPUT_NAME dmk EXPORT_NAME dmk
  VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

target_link_libraries(dmk INTERFACE ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_GPU_TARGET}
  ${DMK_CUDA_JIT_TARGETS} ${DMK_JIT_TARGET} ${DMK_GPU_ESP_INTERFACE} BLAS::BLAS)
target_link_libraries(dmk_shared INTERFACE ${DMK_MPI_TARGET})

set(DMK_BUNDLE_TARGETS "")
foreach(dep finufft finufft_common cufinufft ducc0 nda_c rufus)
  if (TARGET ${dep})
    get_target_property(dep_type ${dep} TYPE)
    if (dep_type STREQUAL "STATIC_LIBRARY")
      list(APPEND DMK_BUNDLE_TARGETS ${dep})
    elseif (dep_type STREQUAL "SHARED_LIBRARY")
      message(FATAL_ERROR "${dep} is shared, so it cannot be archived into libdmk.a. "
        "Configure with -DBUILD_SHARED_LIBS=OFF, or delete CMakeCache.txt if it holds a stale ON.")
    endif()
  endif()
endforeach()
if (DMK_BUNDLE_TARGETS)
  set(DMK_BUNDLE_FILES "")
  foreach(dep ${DMK_BUNDLE_TARGETS})
    list(APPEND DMK_BUNDLE_FILES $<TARGET_FILE:${dep}>)
  endforeach()
  string(REPLACE ";" "|" DMK_BUNDLE_FILES_ARG "${DMK_BUNDLE_FILES}")
  add_dependencies(dmk ${DMK_BUNDLE_TARGETS})
  # add_dependencies is order-only; this is what forces the re-archive.
  set_property(TARGET dmk APPEND PROPERTY LINK_DEPENDS ${DMK_BUNDLE_FILES})
  add_custom_command(TARGET dmk POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DDMK_AR=${CMAKE_AR}
      -DDMK_APPLE=${APPLE}
      -DDMK_OUTPUT=$<TARGET_FILE:dmk>
      "-DDMK_INPUTS=${DMK_BUNDLE_FILES_ARG}"
      -P ${PROJECT_SOURCE_DIR}/cmake/bundle_static_libs.cmake
    COMMENT "Bundling static dependencies into libdmk.a"
    VERBATIM
  )
endif()

install(TARGETS dmk dmk_shared EXPORT dmkTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES include/dmk.h ${DMK_GENERATED_INCLUDE_DIR}/dmk/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmk)
install(FILES LICENSE THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/dmk)

install(EXPORT dmkTargets NAMESPACE dmk:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmk)
configure_package_config_file(cmake/dmkConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/dmkConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmk)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/dmkConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dmkConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/dmkConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmk)

set(DMK_PC_CFLAGS "")
set(DMK_PC_LIBS_PRIVATE "")
if (DMK_HAVE_MPI)
  list(APPEND DMK_PC_CFLAGS "-DDMK_HAVE_MPI")
  foreach(dir ${MPI_CXX_INCLUDE_DIRS})
    list(APPEND DMK_PC_CFLAGS "-I${dir}")
  endforeach()
  list(APPEND DMK_PC_LIBS_PRIVATE ${MPI_CXX_LIBRARIES})
endif()
if (DMK_HAVE_OPENMP)
  list(APPEND DMK_PC_LIBS_PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()
if (DMK_GPU_OFFLOAD)
  list(APPEND DMK_PC_LIBS_PRIVATE "-L${CUDAToolkit_LIBRARY_DIR}" -lcudart -lcuda -lnvrtc -lcufft)
endif()
if (DMK_USE_JIT)
  list(APPEND DMK_PC_LIBS_PRIVATE "-L${LLVM_LIBRARY_DIR}" -lLLVM)
endif()
list(APPEND DMK_PC_LIBS_PRIVATE ${BLAS_LIBRARIES})
string(REPLACE ";" " " DMK_PC_CFLAGS "${DMK_PC_CFLAGS}")
string(REPLACE ";" " " DMK_PC_LIBS_PRIVATE "${DMK_PC_LIBS_PRIVATE}")

if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(DMK_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
else()
  set(DMK_PC_LIBDIR "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
endif()
if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  set(DMK_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  set(DMK_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
configure_file(cmake/dmk.pc.in ${CMAKE_CURRENT_BINARY_DIR}/dmk.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dmk.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if (DMK_BUILD_TESTS)
  add_subdirectory(test)
endif()

if (DMK_BUILD_EXAMPLES)
  add_subdirectory(examples)
  add_subdirectory(scripts)
endif()
