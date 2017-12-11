
add_library(cereal INTERFACE)
target_include_directories(cereal INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/cereal-1.2.2/include"
)

include_directories(glew)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)


