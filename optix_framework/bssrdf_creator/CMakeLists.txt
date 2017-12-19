
set(SRCS    
bssrdf_creator.cpp
bssrdf_loader.cpp
bssrdf_parameter_manager.cpp
full_bssrdf_generator.cpp
main.cpp
reference_bssrdf.cpp
reference_bssrdf_gpu.cpp
)

set(CUDA_SOURCES    
CUDA_files/planar_bssrdf.cu
CUDA_files/reference_bssrdf.cu
CUDA_files/reference_bssrdf_gpu.cu
CUDA_files/render_bssrdf.cu
)

set(HEADERS
bssrdf_creator.h
bssrdf_loader.h
bssrdf_parameter_manager.h
full_bssrdf_generator.h
full_bssrdf_host_device_common.h
photon_trace_reference_bssrdf.h
photon_trace_structs.h
reference_bssrdf.h
reference_bssrdf_gpu.h
)
set( APP_EXE bssrdf_creator )

source_group("Headers" FILES ${HEADERS})
source_group("Source Files" FILES ${SRCS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../optix_framework)
generate_ptx(framework ${HEADERS} ${CUDA_SOURCES} ${OPTIONS})
    
add_executable( ${APP_EXE}
                ${SRCS} ${HEADERS} ${CUDA_SOURCES}  ${generated}) 

find_package(OpenGL REQUIRED)               
target_link_libraries(${APP_EXE} framework optprops gel nfd optix ${CUDA_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} cereal ${XercesC_LIBRARIES} )

macro(copy_dir target folder dest)
add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${folder}
        ${dest})
endmacro(copy_dir)

copy_dir(bssrdf_creator "${PROJECT_SOURCE_DIR}/externals/OptiX/bin64" ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
copy_dir(bssrdf_creator "${PROJECT_SOURCE_DIR}/externals/xercesc/dll" ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
copy_dir(bssrdf_creator "${PROJECT_SOURCE_DIR}/externals/glew/dll" ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
copy_dir(bssrdf_creator "${SAMPLES_PTX_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/PTX_files")
copy_dir(bssrdf_creator "${SAMPLES_PTX_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/PTX_files")

