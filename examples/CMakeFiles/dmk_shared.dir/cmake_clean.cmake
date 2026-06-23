file(REMOVE_RECURSE
  "libdmk.pdb"
  "libdmk.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX Fortran)
  include(CMakeFiles/dmk_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
