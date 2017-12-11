# At LEAST 2.8 but newer is better
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(ray_engine VERSION 0.1 LANGUAGES CXX)

# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

add_library(xercesc STATIC IMPORTED)
set_property(TARGET xercesc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/externals/xercesc)
set_property(TARGET xercesc PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/externals/xercesc/lib64/xerces-c_3.lib)

add_library(glew STATIC IMPORTED)
set_property(TARGET glew PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/externals/glew)
set_property(TARGET glew PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/externals/glew/lib64/glew32.lib)


option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)
find_package(CUDA 5.0 REQUIRED)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

message(WARNING ${OptiX_INSTALL_DIR} )
find_package(OptiX REQUIRED)
include_directories(
  ${OptiX_INCLUDE}
  ${OptiX_INCLUDE}/optixu
  ${CUDA_INCLUDE_DIRS}
  )
  
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_50,code=sm_50
    )
    
set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

function(generate_ptx target_name)
  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  message(WARNING ${source_files})
  # Create the rules to build the PTX from the CUDA files.
  CUDA_WRAP_SRCS( "" PTX generated_files ${source_files} ${cmake_options}
    OPTIONS ${options} )
    
  set(generated ${generated_files} PARENT_SCOPE)
endfunction()  

# Targets that we develop here
add_subdirectory(externals)
add_subdirectory(optprops)
add_subdirectory(nfd)
add_subdirectory(gel)
add_subdirectory(optix_framework)
add_subdirectory(renderer)
add_subdirectory(bssrdf_creator)
