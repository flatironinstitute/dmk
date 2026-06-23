file(REMOVE_RECURSE
  "libdmk_ref.a"
  "libdmk_ref.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX Fortran)
  include(CMakeFiles/dmk_ref.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
