cmake_minimum_required(VERSION 3.16)
project(FingerprintMatching CXX)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Library: fingerprint_core ----
add_library(fingerprint_core
    src/enhancement.cpp
    src/feature_detection.cpp
    src/matching.cpp
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
    # target_compile_definitions(fingerprint_core PUBLIC FP_DEBUG_VIS_FREQ)
    message(STATUS ">>> Debug visualization macro (FP_DEBUG_VIS) is ENABLED")
endif()

# ---- Others ----
add_subdirectory(test)