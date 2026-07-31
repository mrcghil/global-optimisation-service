# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(goss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

find_path(CPPAD_INCLUDE_DIR cppad/cppad.hpp
  HINTS $ENV{CPPAD_HOME}/include /usr/include /usr/local/include /opt/homebrew/include)
find_path(CPPADCG_INCLUDE_DIR cppad/cg.hpp
  HINTS $ENV{CPPADCG_HOME}/include /usr/include /usr/local/include /opt/homebrew/include)
if(NOT CPPAD_INCLUDE_DIR OR NOT CPPADCG_INCLUDE_DIR)
  message(FATAL_ERROR "CppAD/CppADCodeGen headers not found. Set CPPAD_HOME / CPPADCG_HOME.")
endif()

find_library(CPPAD_LIB cppad_lib
  HINTS /usr/local/lib /usr/lib /opt/homebrew/lib)
# cppad_lib ships CppAD::local::temp_file and other non-template utilities
# needed by the CppADCodeGen JIT pipeline; it is not required for header-only
# CppAD usage but IS required once we use ADFun::optimize() and CodeGen.
if(NOT CPPAD_LIB)
  message(WARNING "libcppad_lib not found; JIT compilation may fail at link time.")
endif()

add_library(cppadcg INTERFACE)
target_include_directories(cppadcg INTERFACE ${CPPAD_INCLUDE_DIR} ${CPPADCG_INCLUDE_DIR})
# Link dl (for dlopen) and cppad_lib (for CppAD runtime utilities used by CodeGen).
target_link_libraries(cppadcg INTERFACE ${CMAKE_DL_LIBS} $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>)

enable_testing()
add_library(goss_ad INTERFACE)
target_include_directories(goss_ad INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Non-template compile+load plumbing for CppADCGBackend (Task 7).
# Built as a static library so that the JIT pipeline code is compiled
# only once, keeping the main tests target lean.
add_library(goss_ad_impl STATIC src/ad/cppadcg_backend.cpp)
target_include_directories(goss_ad_impl PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_ad_impl PUBLIC cppadcg)

add_executable(goss_ad_tests
  tests/ad/test_smoke.cpp
  tests/ad/test_fixtures.cpp
  tests/ad/test_verification.cpp
  tests/ad/test_ad_backend_contract.cpp
  tests/ad/test_cppadcg_link.cpp
  tests/ad/test_cppadcg_backend.cpp)
target_link_libraries(goss_ad_tests PRIVATE goss_ad goss_ad_impl cppadcg GTest::gtest_main)
target_include_directories(goss_ad_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
include(GoogleTest)
gtest_discover_tests(goss_ad_tests)
