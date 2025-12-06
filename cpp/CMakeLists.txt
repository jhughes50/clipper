cmake_minimum_required(VERSION 3.0.2)
project(clipper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(Json_INCLUDE_DIRS "/usr/include/jsoncpp")
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
set(TORCH_ROOT "/opt/libtorch")

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
find_package(Torch REQUIRED PATHS "${TORCH_ROOT}")
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
    include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
    src/tokenizer.cpp
    src/processor_mixins.cpp
    src/processor.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    yaml-cpp
)

enable_testing()
add_executable(merge_test test/test_merge.cpp)
target_link_libraries(merge_test
    GTest::GTest
    GTest::Main
    ${PROJECT_NAME}
)

add_executable(vocab_test test/test_vocab.cpp)
target_link_libraries(vocab_test
    GTest::GTest
    GTest::Main
    ${PROJECT_NAME}
)

add_executable(tokenizer_test test/test_tokenizer.cpp)
target_link_libraries(tokenizer_test
    GTest::GTest
    GTest::Main
    ${PROJECT_NAME}
)

add_executable(processor_mixins_test test/test_processor_mixins.cpp)
target_link_libraries(processor_mixins_test
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(normalize_test test/test_normalize.cpp)
target_link_libraries(normalize_test
    ${OpenCV_LIBRARIES}
    ${PROJECT_NAME}
)

add_executable(processor_test test/test_processor.cpp)
target_link_libraries(processor_test
    GTest::GTest
    GTest::Main
    ${PROJECT_NAME}
)

include(GoogleTest)
gtest_discover_tests(merge_test)
gtest_discover_tests(vocab_test)
gtest_discover_tests(tokenizer_test)
gtest_discover_tests(procerssor_mixins_test)
gtest_discover_tests(processor_test)
