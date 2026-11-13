# Standalone smoke test for the installed package: header search path, exported target and link
# closure. Configure it against an install prefix with -DCMAKE_PREFIX_PATH, never as a subdirectory
# of the DMK build, so it cannot reach the in-tree dmk_internal.
cmake_minimum_required(VERSION 3.28)
project(dmk_consumer LANGUAGES C CXX)

find_package(dmk REQUIRED)

add_executable(dmk_consumer consumer.c)
target_link_libraries(dmk_consumer PRIVATE dmk::dmk)
