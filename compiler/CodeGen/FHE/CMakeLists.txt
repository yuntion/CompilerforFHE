add_library(CrossFHECodeGen CodeGen.cpp)
target_include_directories(CrossFHECodeGen PUBLIC "${PROJECT_SOURCE_DIR}/compiler")
target_link_libraries(CrossFHECodeGen PUBLIC CrossFHEHEIR
  MLIRFuncDialect MLIRArithDialect MLIRIR LLVMSupport)
if(NOT MSVC)
  target_compile_options(CrossFHECodeGen PRIVATE -fno-rtti)
endif()
