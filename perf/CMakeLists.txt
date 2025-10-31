add_executable(direct_kernel_bench direct_kernel_bench.cpp
  ${PROJECT_SOURCE_DIR}/src/prolate0_eval.cpp
  ${PROJECT_SOURCE_DIR}/src/prolate.cpp
  ${PROJECT_SOURCE_DIR}/src/legeexps.cpp)

target_include_directories(direct_kernel_bench PRIVATE  ${DMK_INCLUDES})
target_link_libraries(direct_kernel_bench nanobench polyfit::polyfit)
