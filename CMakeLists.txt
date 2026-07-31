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

enable_testing()
add_library(goss_ad INTERFACE)
target_include_directories(goss_ad INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_executable(goss_ad_tests tests/ad/test_smoke.cpp tests/ad/test_fixtures.cpp tests/ad/test_verification.cpp tests/ad/test_ad_backend_contract.cpp)
target_link_libraries(goss_ad_tests PRIVATE goss_ad GTest::gtest_main)
target_include_directories(goss_ad_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
include(GoogleTest)
gtest_discover_tests(goss_ad_tests)
