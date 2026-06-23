file(REMOVE_RECURSE
  "libdmk.a"
  "libdmk.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX Fortran)
  include(CMakeFiles/dmk.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
