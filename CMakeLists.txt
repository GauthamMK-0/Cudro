cmake_minimum_required(VERSION 3.16)
project(cudro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 QUIET)
find_library(TCC_LIBRARY tcc)

include(FetchContent)
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

add_library(cudro_core STATIC
  src/diagnostic.cpp
  src/token.cpp
  src/lexer.cpp
  src/ast.cpp
  src/parser.cpp
  src/sema.cpp
  src/dag.cpp
  src/lower.cpp
  src/ad.cpp
  src/codegen_c.cpp
  src/jit_tcc.cpp
  src/codegen_llvm.cpp
  src/codegen_cuda.cpp
)
target_include_directories(cudro_core PUBLIC include)

add_executable(cudro src/main.cpp)
target_include_directories(cudro PRIVATE include)
target_link_libraries(cudro PRIVATE cudro_core)

if(TARGET Eigen3::Eigen)
  target_link_libraries(cudro PRIVATE Eigen3::Eigen)
  target_compile_definitions(cudro PRIVATE CUDRO_HAS_EIGEN)
endif()

if(TCC_LIBRARY)
  target_link_libraries(cudro_core PRIVATE ${TCC_LIBRARY})
  target_compile_definitions(cudro_core PRIVATE CUDRO_HAS_TCC)
  target_link_libraries(cudro PRIVATE ${TCC_LIBRARY})
  target_compile_definitions(cudro PRIVATE CUDRO_HAS_TCC)
endif()

enable_testing()
add_executable(smoke tests/smoke.cpp)
target_include_directories(smoke PRIVATE include)
add_test(NAME smoke COMMAND smoke)

add_executable(test_lexer tests/test_lexer.cpp)
target_link_libraries(test_lexer PRIVATE cudro_core doctest)
add_test(NAME lexer COMMAND test_lexer)

add_executable(test_parser tests/test_parser.cpp)
target_link_libraries(test_parser PRIVATE cudro_core doctest)
target_compile_definitions(test_parser PRIVATE CUDRO_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME parser COMMAND test_parser)

add_executable(test_ad tests/test_ad.cpp)
target_link_libraries(test_ad PRIVATE cudro_core doctest)
add_test(NAME ad COMMAND test_ad)

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE cudro_core doctest)
add_test(NAME kernels COMMAND test_kernels)

add_executable(test_sema tests/test_sema.cpp)
target_link_libraries(test_sema PRIVATE cudro_core doctest)
add_test(NAME sema COMMAND test_sema)

add_executable(test_dag tests/test_dag.cpp)
target_link_libraries(test_dag PRIVATE cudro_core doctest)
add_test(NAME dag COMMAND test_dag)

add_executable(fuzz_spec tests/fuzz_spec.cpp)
target_link_libraries(fuzz_spec PRIVATE cudro_core doctest)
add_test(NAME fuzz COMMAND fuzz_spec)

option(CUDRO_SANITIZERS "Build tests with ASan+UBSan" ON)
if(CUDRO_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(t smoke test_lexer test_parser test_sema test_dag test_ad test_kernels fuzz_spec)
    target_compile_options(${t} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${t} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()
