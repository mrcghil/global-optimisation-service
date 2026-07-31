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
# Link dl (for dlopen) only; cppad_lib is NOT propagated here because cppadcg
# is header-only and not all consumers instantiate CppAD recording machinery.
# cppad_lib is linked where it is actually needed (see goss_ad_impl and goss_ad_tests).
target_link_libraries(cppadcg INTERFACE ${CMAKE_DL_LIBS})

enable_testing()
add_library(goss_ad INTERFACE)
target_include_directories(goss_ad INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Non-template compile+load plumbing for CppADCGBackend (Task 7).
# Built as a static library so that the JIT pipeline code is compiled
# only once, keeping the main tests target lean.
# cppad_lib is linked PUBLIC so any consumer of goss_ad_impl (e.g. goss_ad_tests
# via the backend) transitively gets the CppAD runtime symbols.
add_library(goss_ad_impl STATIC src/ad/cppadcg_backend.cpp)
target_include_directories(goss_ad_impl PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_ad_impl PUBLIC cppadcg $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>)

add_executable(goss_ad_tests
  tests/ad/test_smoke.cpp
  tests/ad/test_fixtures.cpp
  tests/ad/test_verification.cpp
  tests/ad/test_ad_backend_contract.cpp
  tests/ad/test_cppadcg_link.cpp
  tests/ad/test_cppadcg_backend.cpp)
# test_cppadcg_link.cpp instantiates CppAD::Independent / ADFun<CGD> directly in
# the test TU, so cppad_lib must be linked here in addition to what goss_ad_impl
# already pulls in transitively.
target_link_libraries(goss_ad_tests PRIVATE goss_ad goss_ad_impl cppadcg $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
target_include_directories(goss_ad_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
include(GoogleTest)
gtest_discover_tests(goss_ad_tests)

# ---- NLP core layer ----
add_library(goss_nlp STATIC src/nlp/nlp_problem.cpp)
target_include_directories(goss_nlp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_nlp PUBLIC goss_ad)

add_executable(goss_nlp_tests
  tests/nlp/test_nlp_construction.cpp
  tests/nlp/test_nlp_eval.cpp
  tests/nlp/test_nlp_integration.cpp)
target_link_libraries(goss_nlp_tests PRIVATE
  goss_nlp goss_ad goss_ad_impl cppadcg $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
target_include_directories(goss_nlp_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
gtest_discover_tests(goss_nlp_tests)

# ---- Solver layer: IPOPT + NLopt via pkg-config ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED ipopt)
pkg_check_modules(NLOPT REQUIRED nlopt)

# Reusable INTERFACE targets so every consumer (tests now, goss_solver in the
# solver layer) inherits the correct include dirs, -DHAVE_CSTDDEF, and link
# flags without duplication. IPOPT 3.11.9 headers #error without HAVE_CSTDDEF.
add_library(goss_ipopt_iface INTERFACE)
target_include_directories(goss_ipopt_iface INTERFACE ${IPOPT_INCLUDE_DIRS})
target_compile_options(goss_ipopt_iface INTERFACE ${IPOPT_CFLAGS_OTHER})
target_link_libraries(goss_ipopt_iface INTERFACE ${IPOPT_LIBRARIES})
target_link_directories(goss_ipopt_iface INTERFACE ${IPOPT_LIBRARY_DIRS})

add_library(goss_nlopt_iface INTERFACE)
target_include_directories(goss_nlopt_iface INTERFACE ${NLOPT_INCLUDE_DIRS})
target_compile_options(goss_nlopt_iface INTERFACE ${NLOPT_CFLAGS_OTHER})
target_link_libraries(goss_nlopt_iface INTERFACE ${NLOPT_LIBRARIES})
target_link_directories(goss_nlopt_iface INTERFACE ${NLOPT_LIBRARY_DIRS})

add_library(goss_solver STATIC src/solver/ipopt_solver.cpp src/solver/nlopt_solver.cpp)
target_include_directories(goss_solver PUBLIC ${CMAKE_SOURCE_DIR}/include)
# Task 1 created goss_ipopt_iface / goss_nlopt_iface INTERFACE targets that carry
# the include dirs, -DHAVE_CSTDDEF (IPOPT 3.11.9 headers #error without it), and
# link flags. Link them PRIVATE so goss_solver inherits everything in one line.
target_link_libraries(goss_solver PUBLIC goss_nlp goss_ad
  PRIVATE goss_ipopt_iface goss_nlopt_iface)

add_executable(goss_solver_tests
  tests/solver/test_solver_link.cpp
  tests/solver/test_solver_interface.cpp
  tests/solver/test_ipopt_solver.cpp
  tests/solver/test_nlopt_solver.cpp
  tests/solver/test_solver_agreement.cpp)
target_include_directories(goss_solver_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(goss_solver_tests PRIVATE
  goss_solver goss_ipopt_iface goss_nlopt_iface
  goss_ad_impl cppadcg $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>
  GTest::gtest_main)
gtest_discover_tests(goss_solver_tests)

# ---- Transcription layer ----
add_library(goss_transcription STATIC
  src/transcription/trapezoidal.cpp
  src/transcription/hermite_simpson.cpp)
target_include_directories(goss_transcription PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(goss_transcription PUBLIC goss_nlp goss_ad
  PRIVATE goss_ad_impl cppadcg $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}>)

add_executable(goss_transcription_tests
  tests/transcription/test_variable_layout.cpp
  tests/transcription/test_trapezoidal.cpp)
target_include_directories(goss_transcription_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(goss_transcription_tests PRIVATE
  goss_transcription goss_nlp goss_ad goss_ad_impl goss_solver
  goss_ipopt_iface goss_nlopt_iface cppadcg
  $<$<BOOL:${CPPAD_LIB}>:${CPPAD_LIB}> GTest::gtest_main)
gtest_discover_tests(goss_transcription_tests)
