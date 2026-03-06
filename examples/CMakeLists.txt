if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example dmk ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})

  if(DMK_BUILD_PVFMM)
    add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
    target_link_libraries(compare_dmk_pvfmm dmk pvfmmStatic ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})
  endif()
endif()

if (DMK_HAVE_MPI AND DMK_BUILD_PVFMM)
    add_executable(debug_mpi debug_mpi.cpp)
    target_link_libraries(debug_mpi dmk pvfmmStatic ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})
endif()
