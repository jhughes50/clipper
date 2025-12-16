cmake_minimum_required(VERSION 3.0.2)
project(clipper VERSION 1.0.0)

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
find_package(glog REQUIRED)

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
    src/clipper_model.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    glog::glog
    yaml-cpp
)

# Set include directories for the library target
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
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

add_executable(model_test test/test_model.cpp)
target_link_libraries(model_test
    ${PROJECT_NAME}
)

add_executable(model_inference_test test/test_model_inference.cpp)
target_link_libraries(model_inference_test
    ${PROJECT_NAME}
)

include(GoogleTest)
gtest_discover_tests(merge_test)
gtest_discover_tests(vocab_test)
gtest_discover_tests(tokenizer_test)
gtest_discover_tests(processor_mixins_test)
gtest_discover_tests(processor_test)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library
install(TARGETS ${PROJECT_NAME}
    EXPORT ClipperTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install the export set
install(EXPORT ClipperTargets
    FILE ClipperTargets.cmake
    NAMESPACE Clipper::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Clipper
)

# Create config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ClipperConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ClipperConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Clipper
)

# Create version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ClipperConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ClipperConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ClipperConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Clipper
)
