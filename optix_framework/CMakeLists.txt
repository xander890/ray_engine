# At LEAST 2.8 but newer is better
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(ray_engine)

# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )

if(WIN32)
  set (CMAKE_CXX_STANDARD 11)
  add_definitions(/bigobj)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  message(STATUS "Flags ${CMAKE_CXX_FLAGS}")
else()
    set (CMAKE_CXX_STANDARD 17)
    add_compile_options(-fpermissive)
    FIND_PACKAGE(PkgConfig REQUIRED)
    PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)
    INCLUDE_DIRECTORIES(${GTK3_INCLUDE_DIRS})
    LINK_DIRECTORIES(${GTK3_LIBRARY_DIRS})
    add_definitions (${GTK3_CFLAGS_OTHER})
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
endif()

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/cereal-1.2.2/include
)

# Importing libraries.
if(UNIX)
find_package(DevIL)
else()
set(IL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/devil/include)
set(IL_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/externals/devil/lib/DevIL.lib)
set(ILU_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/externals/devil/lib/ILU.lib)
set(ILUT_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/externals/devil/lib/ILUT.lib)
endif()

include_directories(
        ${IL_INCLUDE_DIR}
)

if(UNIX)
find_package(GLEW)
else()
set(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/externals/glew)
set(GLEW_LIBRARIES  ${CMAKE_CURRENT_SOURCE_DIR}/externals/glew/lib64/glew32.lib)
endif()

include_directories(
        ${GLEW_INCLUDE_DIRS}
)

option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)
find_package(CUDA 9.0 REQUIRED)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

find_package(OptiX 5.0 REQUIRED)
include_directories(
  ${OptiX_INCLUDE}
  ${OptiX_INCLUDE}/optixu
  ${CUDA_INCLUDE_DIRS}
  )

find_package(Eigen3 REQUIRED)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  )
  
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_60,code=sm_60
    )
    
set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

function(generate_ptx target_name)
  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Create the rules to build the PTX from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options}
    OPTIONS ${options})
    
  set(generated ${generated_files} PARENT_SCOPE)
endfunction()  

# Targets that we develop here
add_subdirectory(externals)
add_subdirectory(optprops)
add_subdirectory(nfd)
add_subdirectory(optix_framework)
add_subdirectory(renderer)
#add_subdirectory(bssrdf_creator)
