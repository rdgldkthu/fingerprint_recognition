cmake_minimum_required(VERSION 3.16)
project(FingerprintMatching CXX)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Library: fingerprint_core ----
add_library(fingerprint_core
    src/core/frequency.cpp
    src/core/gabor.cpp
    src/core/minutiae.cpp
    src/core/orientation.cpp
    src/core/roi.cpp
    src/core/skeleton.cpp
    src/core/utility.cpp
    src/enhancement/enhancement.cpp
    src/features/detection.cpp
    src/features/mcc.cpp
)

target_link_libraries(fingerprint_core ${OpenCV_LIBS})
target_include_directories(fingerprint_core PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- Executable ----
add_executable(fingerprint_app src/main.cpp)
target_link_libraries(fingerprint_app PRIVATE fingerprint_core)

# ---- Macros ----
target_compile_definitions(fingerprint_core PUBLIC
    DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

# --- Debug Visualizations ----
option(ENABLE_DEBUG_VIS "Enable visualization for debugging" OFF)
if(ENABLE_DEBUG_VIS)
    target_compile_definitions(fingerprint_core PUBLIC FP_DEBUG_VIS)
    message(STATUS ">>> Debug visualization macro (FP_DEBUG_VIS) is ENABLED")
endif()

# ---- Others ----
add_subdirectory(test)