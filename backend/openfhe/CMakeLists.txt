find_package(OpenFHE 1.4.2 EXACT REQUIRED CONFIG)
add_executable(model-runner runner.cpp)
target_include_directories(model-runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${OpenFHE_INCLUDE}" "${OpenFHE_INCLUDE}/core"
  "${OpenFHE_INCLUDE}/pke" "${OpenFHE_INCLUDE}/binfhe")
target_compile_definitions(model-runner PRIVATE MATHBACKEND=${OpenFHE_BACKEND})
target_link_libraries(model-runner PRIVATE ${OpenFHE_LIBRARIES} ${OPENMP_LIBRARIES})
