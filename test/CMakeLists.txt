add_executable(test_all test_all.cpp test_pbc_periodic.cpp test_direct.cpp test_esp.cpp)
target_link_libraries(test_all PRIVATE dmk_shared)
target_compile_definitions(test_all PRIVATE
    VERIFY_SCRIPT_PATH="${CMAKE_SOURCE_DIR}/examples/verify_esp.py"
    PERILAP3D_DIR="${CMAKE_SOURCE_DIR}/scripts/perilap3d"
)
