cmake_minimum_required(VERSION 3.16)
project(FingerprintMatching CXX)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Include directories ----
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# ---- Library: fingerprint_core ----
add_library(fingerprint_core
    src/enhancement.cpp
    src/feature_detection.cpp
    src/matching.cpp
)

target_link_libraries(fingerprint_core
    ${OpenCV_LIBS}
)

target_include_directories(fingerprint_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# ---- Executable: main pipeline ----
add_executable(fingerprint_app
    src/main.cpp
)

target_link_libraries(fingerprint_app
    fingerprint_core
)

# ---- Tests ----
# option(BUILD_TESTS "Build tests" ON)

# if(BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(tests)
# endif()

# ---- Macros ----
add_definitions(-DDATA_DIR="${CMAKE_SOURCE_DIR}/data")
# add_definitions(-DFP_DEBUG_VIS)

# ---- Others ----
add_subdirectory(test)