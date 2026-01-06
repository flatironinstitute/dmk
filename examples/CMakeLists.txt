if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example dmk ${PAPI_TARGET} ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET})

  if(DMK_BUILD_PVFMM)
    add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
    target_link_libraries(compare_dmk_pvfmm dmk ${PAPI_TARGET} pvfmmStatic ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET})
  endif()
endif()
