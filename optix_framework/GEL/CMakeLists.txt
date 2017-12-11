include_directories(CGLA)
include_directories(GLGraphics)
include_directories(Util)

set(HEADERS
gl_wrap.h
CGLA/ArithMatFloat.h
CGLA/ArithQuat.h
CGLA/ArithSqMat2x2Float.h
CGLA/ArithSqMat3x3Float.h
CGLA/ArithSqMat4x4Float.h
CGLA/ArithSqMatFloat.h
CGLA/ArithVec.h
CGLA/ArithVec2Float.h
CGLA/ArithVec3Float.h
CGLA/ArithVec3Int.h
CGLA/ArithVec4Float.h
CGLA/ArithVec4Int.h
CGLA/ArithVecFloat.h
CGLA/ArithVecInt.h
CGLA/BitMask.h
CGLA/CGLA.h
CGLA/eigensolution.h
CGLA/ExceptionStandard.h
CGLA/Mat2x2d.h
CGLA/Mat2x2f.h
CGLA/Mat2x3d.h
CGLA/Mat2x3f.h
CGLA/Mat3x3d.h
CGLA/Mat3x3f.h
CGLA/Mat4x4d.h
CGLA/Mat4x4f.h
CGLA/Quatd.h
CGLA/Quaternion.h
CGLA/Quatf.h
CGLA/statistics.h
CGLA/TableTrigonometry.h
CGLA/UnitVector.h
CGLA/Vec2d.h
CGLA/Vec2f.h
CGLA/Vec2i.h
CGLA/Vec2ui.h
CGLA/Vec3d.h
CGLA/Vec3f.h
CGLA/Vec3Hf.h
CGLA/Vec3i.h
CGLA/Vec3uc.h
CGLA/Vec3usi.h
CGLA/Vec4d.h
CGLA/Vec4f.h
CGLA/Vec4i.h
CGLA/Vec4uc.h
GLGraphics/gel_gl.h
GLGraphics/gel_glu.h
GLGraphics/gel_glut.h
GLGraphics/glsl_shader.h
GLGraphics/image_DXT.h
GLGraphics/image_helper.h
GLGraphics/SinglePassWireframeRenderer.h
GLGraphics/SOIL.h
GLGraphics/stbi_DDS_aug.h
GLGraphics/stbi_DDS_aug_c.h
GLGraphics/stb_image.h
GLGraphics/stb_image_write.h
Util/ArgExtracter.h
Util/Grid2D.h
Util/HashKey.h
Util/HashTable.h
Util/Parse.h
Util/ResourceManager.h
Util/string_utils.h
Util/Timer.h
Util/XmlParser.h
)

set(SRCS    CGLA/ArithSqMat3x3Float.cpp
            CGLA/ArithSqMat4x4Float.cpp
            CGLA/ArithVec2Float.cpp
            CGLA/ArithVec3Float.cpp
            CGLA/eigensolution.cpp
            CGLA/gel_rand.cpp
            CGLA/Mat3x3d.cpp
            CGLA/Mat3x3f.cpp
            CGLA/Mat4x4d.cpp
            CGLA/Mat4x4f.cpp
            CGLA/statistics.cpp
            CGLA/TableTrigonometry.cpp
            CGLA/Vec2i.cpp
            CGLA/Vec3f.cpp
            CGLA/Vec3i.cpp
            GLGraphics/glsl_shader.cpp
            GLGraphics/SinglePassWireframeRenderer.cpp
            Util/HashKey.cpp
            Util/Parse.cpp
            Util/string_utils.cpp
            Util/XmlParser.cpp

    )

source_group("Headers" FILES ${HEADERS})
source_group("Source Files" FILES ${SRCS})
        
ADD_LIBRARY( gel STATIC
             ${SRCS} ${HEADERS})
target_include_directories( gel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(gel glew )            
