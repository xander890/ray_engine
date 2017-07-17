#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"

struct MeshData
{
    optix::Buffer mVbuffer;
    optix::Buffer mNbuffer;
    optix::Buffer mTBuffer;
    optix::Buffer mVIbuffer;
    optix::Buffer mNIbuffer;
    optix::Buffer mTIbuffer;
    optix::Buffer mMatBuffer;
    int mNumTriangles;
};

class Mesh
{
public:
    Mesh(optix::Context ctx);
        
    void init(MeshData meshdata, MaterialData material);

    optix::GeometryInstance mGeometryInstance = nullptr;
    optix::Geometry mGeometry = nullptr;
    optix::Context  mContext;
    optix::Material mMaterial = nullptr;
    MaterialData mMaterialData;
    MeshData mMeshData;
    Shader * mShader;

    void load_material();  
    void load_geometry();
    void load_shader(Mesh& object, RenderingMethodType::EnumType method);

private:
    void create_and_bind_optix_data();
    optix::Program         mIntersectProgram;
    optix::Program         mBoundingboxProgram;

};
