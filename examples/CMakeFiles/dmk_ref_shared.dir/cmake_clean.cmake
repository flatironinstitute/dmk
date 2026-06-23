file(REMOVE_RECURSE
  "libdmk_ref.pdb"
  "libdmk_ref.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX Fortran)
  include(CMakeFiles/dmk_ref_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
