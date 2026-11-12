add_library(CrossFHEConversion LowerToSCFHE.cpp LowerToHEIR.cpp)
target_include_directories(CrossFHEConversion PUBLIC "${PROJECT_SOURCE_DIR}/compiler")
target_link_libraries(CrossFHEConversion PUBLIC CrossFHEDialect
  MLIRPass MLIRFuncDialect MLIRArithDialect MLIRTransforms)
if(NOT MSVC)
  target_compile_options(CrossFHEConversion PRIVATE -fno-rtti)
endif()
