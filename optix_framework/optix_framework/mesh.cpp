#include "mesh.h"
#include "shader_factory.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>
#include<host_material.h>

Mesh::Mesh(optix::Context ctx) : mContext(ctx)
{
}



void Mesh::init(MeshData meshdata, std::shared_ptr<MaterialHost> material)
{
    mMeshData = meshdata;
    mMaterialData = material;
    // Load triangle_mesh programs
    if (!mIntersectProgram.get()) {
        std::string path = std::string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh.cu.ptx";
        mIntersectProgram = mContext->createProgramFromPTXFile(path, "mesh_intersect");
    }

    if (!mBoundingboxProgram.get()) {
        std::string path = std::string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh.cu.ptx";
        mBoundingboxProgram = mContext->createProgramFromPTXFile(path, "mesh_bounds");
    }

    load_geometry();
    load_material();
}

void Mesh::load_material()
{
    create_and_bind_optix_data();
    const auto d = mMaterialData->get_data();
    mMaterial["material"]->setUserData(sizeof(MaterialDataCommon), &d);    
}

void Mesh::load_geometry()
{
    create_and_bind_optix_data();

    mGeometry->setPrimitiveCount(mMeshData.mNumTriangles);
    mGeometry->setIntersectionProgram(mIntersectProgram);
    mGeometry->setBoundingBoxProgram(mBoundingboxProgram);
    mGeometry["vertex_buffer"]->setBuffer(mMeshData.mVbuffer);
    mGeometry["vindex_buffer"]->setBuffer(mMeshData.mVIbuffer);
    mGeometry["normal_buffer"]->setBuffer(mMeshData.mNbuffer);
    mGeometry["nindex_buffer"]->setBuffer(mMeshData.mNIbuffer);
    mGeometry["texcoord_buffer"]->setBuffer(mMeshData.mTBuffer);
    mGeometry["tindex_buffer"]->setBuffer(mMeshData.mTIbuffer);
    mGeometry["material_buffer"]->setBuffer(mMeshData.mMatBuffer);
    mGeometry["num_triangles"]->setUint(mMeshData.mNumTriangles);
    mGeometry->markDirty();
}

void Mesh::load_shader(Mesh& object, RenderingMethodType::EnumType method)
{
    mShader = ShaderFactory::get_shader(mMaterialData->get_data().illum, method);
    mShader->initialize_mesh(object);
}

void Mesh::create_and_bind_optix_data()
{
    bool bind = false;
    if (!mGeometry)
    {
        mGeometry = mContext->createGeometry();
        bind = true;
    }

    if (!mGeometryInstance)
    {
        mGeometryInstance = mContext->createGeometryInstance();
        bind = true;
    }

    if (!mMaterial)
    {
        mMaterial = mContext->createMaterial();
        bind = true;
    }

    if (bind)
    {
        mGeometryInstance->setGeometry(mGeometry);
        mGeometryInstance->setMaterialCount(1);
        mGeometryInstance->setMaterial(0, mMaterial);
    }    
}
