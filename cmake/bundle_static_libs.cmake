# Merge the '|'-separated archives in DMK_INPUTS into the archive DMK_OUTPUT.
string(REPLACE "|" ";" inputs "${DMK_INPUTS}")
set(bundled "${DMK_OUTPUT}.bundle")

if (DMK_APPLE)
  execute_process(COMMAND libtool -static -o "${bundled}" "${DMK_OUTPUT}" ${inputs}
    RESULT_VARIABLE result ERROR_VARIABLE error)
else()
  set(script "create ${bundled}\naddlib ${DMK_OUTPUT}\n")
  foreach(lib ${inputs})
    string(APPEND script "addlib ${lib}\n")
  endforeach()
  string(APPEND script "save\nend\n")
  file(WRITE "${DMK_OUTPUT}.mri" "${script}")
  execute_process(COMMAND "${DMK_AR}" -M INPUT_FILE "${DMK_OUTPUT}.mri"
    RESULT_VARIABLE result ERROR_VARIABLE error)
  file(REMOVE "${DMK_OUTPUT}.mri")
endif()

if (NOT result EQUAL 0)
  message(FATAL_ERROR "failed to bundle static libraries into ${DMK_OUTPUT}: ${error}")
endif()
file(RENAME "${bundled}" "${DMK_OUTPUT}")
