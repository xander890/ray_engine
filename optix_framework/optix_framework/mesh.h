#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"
#include <memory>

class MaterialHost;

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
    explicit Mesh(optix::Context ctx);

    void init(MeshData meshdata, std::shared_ptr<MaterialHost> material);

    optix::GeometryInstance mGeometryInstance = nullptr;
    optix::Geometry mGeometry = nullptr;
    optix::Context  mContext;
    optix::Material mMaterial = nullptr;

    MeshData mMeshData;
    std::shared_ptr<Shader> mShader;

    void load_material();  
    void load_geometry();
    void load_shader(RenderingMethodType::EnumType method);
    void add_material(std::shared_ptr<MaterialHost> material);

    std::shared_ptr<MaterialHost> get_main_material() { return mMaterialData[0]; }

private:
    std::vector<std::shared_ptr<MaterialHost>> mMaterialData;
    void create_and_bind_optix_data();
    optix::Program         mIntersectProgram;
    optix::Program         mBoundingboxProgram;
    optix::Buffer          mMaterialBuffer;

};
