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

add_library(cppadcg INTERFACE)
target_include_directories(cppadcg INTERFACE ${CPPAD_INCLUDE_DIR} ${CPPADCG_INCLUDE_DIR})
target_link_libraries(cppadcg INTERFACE ${CMAKE_DL_LIBS})  # dlopen for JIT

enable_testing()
add_library(goss_ad INTERFACE)
target_include_directories(goss_ad INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_executable(goss_ad_tests
  tests/ad/test_smoke.cpp
  tests/ad/test_fixtures.cpp
  tests/ad/test_verification.cpp
  tests/ad/test_ad_backend_contract.cpp
  tests/ad/test_cppadcg_link.cpp)
target_link_libraries(goss_ad_tests PRIVATE goss_ad cppadcg GTest::gtest_main)
target_include_directories(goss_ad_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
include(GoogleTest)
gtest_discover_tests(goss_ad_tests)
