set(TEST_ALL_SRC test_all.cpp test_pbc_periodic.cpp test_direct.cpp test_esp.cpp)
if (DMK_BUILD_REFERENCE)
  list(APPEND TEST_ALL_SRC test_reference.cpp)
endif()
add_executable(test_all ${TEST_ALL_SRC})
target_link_libraries(test_all PRIVATE dmk_shared)
if (DMK_BUILD_REFERENCE)
  target_link_libraries(test_all PRIVATE dmk_ref)
endif()
target_compile_definitions(test_all PRIVATE
    VERIFY_SCRIPT_PATH="${CMAKE_SOURCE_DIR}/examples/verify_esp.py"
    PERILAP3D_DIR="${CMAKE_SOURCE_DIR}/scripts/perilap3d"
    VERIFY_CACHE_DIR="${CMAKE_BINARY_DIR}"
)
