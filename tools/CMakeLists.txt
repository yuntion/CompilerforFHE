if(NOT CROSSFHE_ENABLE_LEGACY_COMPILER)
  add_executable(crossfhe-opt crossfhe-opt.cpp)
  target_link_libraries(crossfhe-opt PRIVATE CrossFHEPipelines MLIROptLib MLIRParser)
  add_executable(crossfhe-translate crossfhe-translate.cpp)
  target_link_libraries(crossfhe-translate PRIVATE CrossFHECodeGen MLIRParser)
  foreach(target crossfhe-opt crossfhe-translate)
    if(NOT MSVC)
      target_compile_options(${target} PRIVATE -fno-rtti)
    endif()
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  endforeach()
  return()
endif()

# Special case package namespace for tools/ -> "" (empty string)
# For example, tools/nisl-compile -> nisl-compile (no namespace)
set(NISL_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
# /home/MLIR-Compiler/tools:"===================="



set(NISL_PACKAGE_ROOT_PREFIX "Mlir_Compiler/tools")
nisl_add_all_subdirs()
set(NISL_PACKAGE_ROOT_PREFIX "")


# TODO(scotttodd): Should this be checking NISL_BUILD_COMPILER?
#   Maybe we should disallow setting both at the same time, since it's
#   ambigious which should be used
if(NISL_HOST_BIN_DIR AND NOT NISL_BUILD_COMPILER)
  nisl_import_binary(NAME nisl-tblgen)
  nisl_import_binary(NAME nisl-compile)
  nisl_import_binary(NAME nisl-opt)

endif()



if(NISL_BUILD_COMPILER)


  nisl_cc_binary(
    NAME
      nisl-compile
    SRCS
      "nisl-compile-main.cc"
    DEPS
      compilerImpl
    DATA
    HOSTONLY
    SETUP_INSTALL_RPATH 
  )
  nisl_cc_binary(
    NAME
      nisl-opt
    SRCS
      "nisl-opt-main.cc"
    DEPS
      optImpl

  )


  nisl_cc_binary(
    NAME
      nisl-tblgen
    SRCS
      "${NISL_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
      # "${NISL_SOURCE_DIR}/compiler/Dialect/VM/Tools/VMOpEncoderGen.cpp"
      # "${NISL_SOURCE_DIR}/compiler/Dialect/VM/Tools/VMOpTableGen.cpp"
    DEPS
      LLVMSupport
      LLVMTableGen
      MLIRSupport
      MLIRTableGen
      MLIRTblgenLib
    HOSTONLY
  )


endif()
