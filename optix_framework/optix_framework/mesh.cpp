#include "mesh.h"
#include "shader_factory.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>
#include<host_material.h>
#include "optix_utils.h"

Mesh::Mesh(optix::Context ctx) : mContext(ctx)
{
}

void Mesh::init(const char* name, MeshData meshdata, std::shared_ptr<MaterialHost> material)
{
    mMeshName = name;
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
        mMaterialBuffer = create_buffer<MaterialDataCommon>(mContext);
    }

    if (!mBBoxBuffer.get())
    {
        mBBoxBuffer = create_buffer<optix::Aabb>(mContext);
    }

    set_shader(mMaterialData[0]->get_data().illum);
	load();
}

void Mesh::load_materials()
{
	bool one_material_changed = std::any_of(mMaterialData.begin(), mMaterialData.end(), [](const std::shared_ptr<MaterialHost>& mat) { return mat->hasChanged();  });
	mReloadMaterials |= one_material_changed;

	if(!mReloadMaterials)
	{
		return;
	}

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
	mReloadMaterials = false;
}

void Mesh::load_geometry()
{
	if (!mReloadGeometry)
		return;
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
    mGeometryInstance["num_triangles"]->setUint(mMeshData.mNumTriangles);
    initialize_buffer<optix::Aabb>(mBBoxBuffer, mMeshData.mBoundingBox);
    BufPtr<optix::Aabb> bptr = BufPtr<optix::Aabb>(mBBoxBuffer->getId());
    mGeometryInstance["local_bounding_box"]->setUserData(sizeof(BufPtr<optix::Aabb>), &bptr);
    mGeometry->markDirty();
	mReloadGeometry = false;
}

void Mesh::load_shader()
{
	if (mReloadShader || mShader->has_changed())
	{
		mShader->load_into_mesh(*this);
	}
	mReloadShader = false;
}

void Mesh::reload_shader()
{
	mReloadShader = true;
}

void Mesh::load()
{
	load_geometry();
	load_materials();
	load_shader();
}

void Mesh::set_method(RenderingMethodType::EnumType method)
{
    mShader->set_method(method);
	mReloadShader = true;
}

void Mesh::set_shader(int illum)
{
    mShader = ShaderFactory::get_shader(illum);
	mReloadShader = true;
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

void Mesh::setShader(const void* var, void* data)
{
    Mesh* mesh = reinterpret_cast<Mesh*>(data);
    int illum = *(int*)var;
    mesh->set_shader(illum);
}

void Mesh::getShader(void* var, void* data)
{
    Mesh* mesh = reinterpret_cast<Mesh*>(data);
    *((int*)var) = mesh->mShader->get_illum();
}

void Mesh::add_material(std::shared_ptr<MaterialHost> material)
{
    mMaterialData.push_back(material);
    mMaterialBuffer->setSize(mMaterialData.size());
    load_materials();
}

void Mesh::set_into_gui(GUI* gui, const char* group)
{
    std::string nm = mMeshName + "/Rendering Method";
    std::vector<GuiDropdownElement> guiinfo = ShaderFactory::get_gui_info();
    gui->addDropdownMenuCallback(nm.c_str(), guiinfo, setShader, getShader, this, mMeshName.c_str());
	mShader->set_into_gui(gui, group);
    for (auto& m : mMaterialData)
    {
        m->set_into_gui(gui, mMeshName.c_str());
    }
}

void Mesh::remove_from_gui(GUI * gui, const char * group)
{
	std::string nm = mMeshName + "/Rendering Method";
	gui->removeVar(nm.c_str());
	mShader->remove_from_gui(gui, group);
	for (auto& m : mMaterialData)
	{
		m->remove_from_gui(gui, mMeshName.c_str());
	}
}

void Mesh::pre_trace()
{
	mShader->pre_trace_mesh(*this);
}
