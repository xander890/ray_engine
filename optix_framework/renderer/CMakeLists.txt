
set(SRCS    
    main.cpp
    )

set( APP_EXE renderer )
    
ADD_EXECUTABLE( ${APP_EXE}
                ${SRCS} ) 

find_package(OpenGL REQUIRED)               
target_link_libraries(${APP_EXE} framework optprops gel glew nfd optix ${CUDA_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} cereal ${XercesC_LIBRARIES} )

macro(copy_dir target folder dest)
add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${folder}
        ${dest})
endmacro(copy_dir)

if(WIN32)
copy_dir(renderer "${PROJECT_SOURCE_DIR}/externals/OptiX/bin64" ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
copy_dir(renderer "${PROJECT_SOURCE_DIR}/externals/xercesc/dll" ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
copy_dir(renderer "${PROJECT_SOURCE_DIR}/externals/glew/dll" ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
endif()

copy_dir(renderer "${SAMPLES_PTX_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/PTX_files")
copy_dir(renderer "${SAMPLES_PTX_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/PTX_files")

