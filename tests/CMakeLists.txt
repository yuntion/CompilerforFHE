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

add_test(NAME crossfhe-shared-comparisons COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/repeated_relu.mlir"
  --ppml-to-scfhe --ppml-to-heir --ppml-fhe-optimize --verify-each)
set_tests_properties(crossfhe-shared-comparisons PROPERTIES
  PASS_REGULAR_EXPRESSION "ppml.logic_comparisons = 1")
add_test(NAME crossfhe-balanced-multiplication COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/balanced_mul.mlir"
  --ppml-to-scfhe --ppml-to-heir --ppml-fhe-balance-mul --verify-each)
set_tests_properties(crossfhe-balanced-multiplication PROPERTIES
  PASS_REGULAR_EXPRESSION "ppml.multiplicative_depth = 3")
add_test(NAME crossfhe-bounded-relu COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/relu.mlir"
  --ppml-to-scfhe "--ppml-scfhe-approx-relu=bound=1 max-error=0.0625"
  --ppml-to-heir --ppml-fhe-optimize --verify-each)
set_tests_properties(crossfhe-bounded-relu PROPERTIES
  PASS_REGULAR_EXPRESSION "ppml.logic_comparisons = 0")

if(WIN32)
  get_filename_component(CROSSFHE_TOOLCHAIN_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
  set_tests_properties(crossfhe-dialects crossfhe-lowering crossfhe-codegen
    crossfhe-shared-comparisons crossfhe-balanced-multiplication crossfhe-bounded-relu
    PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CROSSFHE_TOOLCHAIN_BIN}")
endif()

add_test(NAME crossfhe-packed-matvec COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/matvec.mlir"
  "--ppml-fhe-matvec=strategy=bsgs" --ppml-fhe-optimize --verify-each)
set_tests_properties(crossfhe-packed-matvec PROPERTIES
  PASS_REGULAR_EXPRESSION "rotations = 2")
if(WIN32)
  set_tests_properties(crossfhe-packed-matvec PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CROSSFHE_TOOLCHAIN_BIN}")
endif()
