# Merge the '|'-separated archives in DMK_INPUTS into the archive DMK_OUTPUT.
string(REPLACE "|" ";" inputs "${DMK_INPUTS}")
set(bundled "${DMK_OUTPUT}.bundle")

if (DMK_APPLE)
  execute_process(COMMAND libtool -static -o "${bundled}" "${DMK_OUTPUT}" ${inputs}
    RESULT_VARIABLE result ERROR_VARIABLE error)
else()
  get_filename_component(output_dir "${DMK_OUTPUT}" DIRECTORY)
  set(stage "${output_dir}/dmk_bundle_stage")
  file(REMOVE_RECURSE "${stage}")
  file(MAKE_DIRECTORY "${stage}")

  set(script "create bundle.a\n")
  set(n 0)
  foreach(lib "${DMK_OUTPUT}" ${inputs})
    file(CREATE_LINK "${lib}" "${stage}/lib${n}.a" SYMBOLIC)
    string(APPEND script "addlib lib${n}.a\n")
    math(EXPR n "${n} + 1")
  endforeach()
  string(APPEND script "save\nend\n")
  file(WRITE "${stage}/bundle.mri" "${script}")

  execute_process(COMMAND "${DMK_AR}" -M INPUT_FILE "${stage}/bundle.mri"
    WORKING_DIRECTORY "${stage}" RESULT_VARIABLE result ERROR_VARIABLE error)
  if (result EQUAL 0)
    file(RENAME "${stage}/bundle.a" "${bundled}")
  endif()
  file(REMOVE_RECURSE "${stage}")
endif()

if (NOT result EQUAL 0)
  message(FATAL_ERROR "failed to bundle static libraries into ${DMK_OUTPUT}: ${error}")
endif()
file(RENAME "${bundled}" "${DMK_OUTPUT}")
