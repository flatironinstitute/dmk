cmake_minimum_required(VERSION 3.1)

project(dmk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options("-march=native")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(DMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
)

add_library(${PROJECT_NAME} STATIC src/dmk.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${DMK_INCLUDES})
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
