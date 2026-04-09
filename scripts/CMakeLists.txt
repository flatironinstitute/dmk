add_executable(generate_aot_kernels generate_aot_kernels.cpp)
target_link_libraries(generate_aot_kernels dmk nda::nda_c spdlog::spdlog_header_only)
target_include_directories(generate_aot_kernels PUBLIC ${DMK_INCLUDES})
