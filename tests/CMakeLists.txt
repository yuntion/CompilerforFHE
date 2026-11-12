if(CROSSFHE_ENABLE_LEGACY_COMPILER)
  nisl_add_all_subdirs()
  return()
endif()

add_test(NAME crossfhe-dialects COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/affine_fhe.mlir" --show-dialects)
set_tests_properties(crossfhe-dialects PROPERTIES PASS_REGULAR_EXPRESSION "scfhe")
add_test(NAME crossfhe-lowering COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/relu.mlir"
  --ppml-to-scfhe --ppml-to-heir --ppml-fhe-optimize --verify-each)
set_tests_properties(crossfhe-lowering PROPERTIES PASS_REGULAR_EXPRESSION "heir.depack")
add_test(NAME crossfhe-codegen COMMAND crossfhe-translate
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/affine_fhe.mlir"
  --backend=seal-ckks -o "${CMAKE_CURRENT_BINARY_DIR}/affine.hpp")

if(WIN32)
  get_filename_component(CROSSFHE_TOOLCHAIN_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
  set_tests_properties(crossfhe-dialects crossfhe-lowering crossfhe-codegen
    PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CROSSFHE_TOOLCHAIN_BIN}")
endif()
