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
    mMaterialData.resize(1);
    mMaterialData[0] = material;
    // Load triangle_mesh programs
    if (!mIntersectProgram.get()) {
        std::string path = std::string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh.cu.ptx";
        mIntersectProgram = mContext->createProgramFromPTXFile(path, "mesh_intersect");
    }

    if (!mBoundingboxProgram.get()) {
        std::string path = std::string(PATH_TO_MY_PTX_FILES) + "/triangle_mesh.cu.ptx";
        mBoundingboxProgram = mContext->createProgramFromPTXFile(path, "mesh_bounds");
    }

    if (!mMaterialBuffer.get())
    {
        mMaterialBuffer = mContext->createBuffer(RT_BUFFER_INPUT);
        mMaterialBuffer->setFormat(RT_FORMAT_USER);
        mMaterialBuffer->setElementSize(sizeof(MaterialDataCommon));
        mMaterialBuffer->setSize(1);
    }

    load_geometry();
    load_material();
}

void Mesh::load_material()
{
    create_and_bind_optix_data();
    std::vector<MaterialDataCommon> data;
    data.resize(mMaterialData.size());
    size_t n = mMaterialData.size();
    for (int i = 0; i < n; i++)
    {
        data[i] = mMaterialData[i]->get_data_copy();
    }
    memcpy(mMaterialBuffer->map(), data.data(), n * sizeof(MaterialDataCommon));
    mMaterialBuffer->unmap();
    mMaterial["material_buffer"]->setBuffer(mMaterialBuffer);    
    mMaterial["main_material"]->setUserData(sizeof(MaterialDataCommon), &data[0]);
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

void Mesh::load_shader(RenderingMethodType::EnumType method)
{
    mShader = std::shared_ptr<Shader>(ShaderFactory::get_shader(mMaterialData[0]->get_data().illum));
    mShader->set_method(method);
    mShader->initialize_mesh(*this);
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

void Mesh::add_material(std::shared_ptr<MaterialHost> material)
{
    mMaterialData.push_back(material);
    mMaterialBuffer->setSize(mMaterialData.size());
    load_material();
}
