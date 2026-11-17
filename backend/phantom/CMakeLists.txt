find_package(CUDAToolkit REQUIRED)
set(CROSSFHE_PHANTOM_SOURCE "" CACHE PATH "Pinned Phantom source tree")
set(CROSSFHE_PHANTOM_LIBRARY "" CACHE FILEPATH "Built Phantom shared library")
if(NOT EXISTS "${CROSSFHE_PHANTOM_SOURCE}/include/phantom.h" OR
   NOT EXISTS "${CROSSFHE_PHANTOM_LIBRARY}")
  message(FATAL_ERROR "Set CROSSFHE_PHANTOM_SOURCE and CROSSFHE_PHANTOM_LIBRARY; bootstrap.py --enable-cuda builds them")
endif()
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CROSSFHE_PHANTOM_SOURCE}/.git")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CROSSFHE_PHANTOM_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE CROSSFHE_PHANTOM_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CROSSFHE_PHANTOM_GIT_RESULT)
  if(NOT CROSSFHE_PHANTOM_GIT_RESULT EQUAL 0 OR
     NOT CROSSFHE_PHANTOM_REVISION STREQUAL "1f4a198443b3af77118e51f53d5b8f332154b875")
    message(FATAL_ERROR "Unsupported Phantom revision; use the pinned 2.1 submodule")
  endif()
endif()
add_library(crossfhe_phantom SHARED IMPORTED)
set_target_properties(crossfhe_phantom PROPERTIES IMPORTED_LOCATION "${CROSSFHE_PHANTOM_LIBRARY}")
set(CROSSFHE_CONTRACT_HEADER "" CACHE FILEPATH
    "Trusted canonical evaluator defining the immutable computation/parameter identity")
set(contract_header "${CROSSFHE_CONTRACT_HEADER}")
if(NOT contract_header)
  set(contract_header "${CROSSFHE_MODEL_HEADER}")
endif()
if(NOT EXISTS "${contract_header}")
  message(FATAL_ERROR "CROSSFHE_CONTRACT_HEADER must identify an existing trusted evaluator")
endif()
# Schedule-only candidates share their verified reference contract, while their
# own source/binary hashes remain separate in the AI validation report.
file(SHA256 "${contract_header}" CROSSFHE_MODEL_ID)
add_executable(model-runner runner.cu)
set_target_properties(model-runner PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(model-runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CROSSFHE_PHANTOM_SOURCE}/include")
target_compile_options(model-runner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
target_compile_definitions(model-runner PRIVATE CROSSFHE_MODEL_ID="${CROSSFHE_MODEL_ID}")
target_link_libraries(model-runner PRIVATE crossfhe_phantom CUDA::cudart)
