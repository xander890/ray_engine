#include "mesh.h"
#include "shader_factory.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>
#include<host_material.h>
#include "optix_utils.h"
#include "immediate_gui.h"
#include <array>
Mesh::Mesh(optix::Context ctx) : mContext(ctx)
{
	static int id = 0;
	mMeshID = id++;
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
    size_t n = mMaterialData.size();
	MaterialDataCommon* data = reinterpret_cast<MaterialDataCommon*>(mMaterialBuffer->map());
    for (int i = 0; i < n; i++)
    {
		memcpy(&data[i], &mMaterialData[i]->get_data(), sizeof(MaterialDataCommon));
    }
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
    mGeometry["num_triangles"]->setUint(mMeshData.mNumTriangles);
    mGeometryInstance["num_triangles"]->setUint(mMeshData.mNumTriangles);
	mGeometryInstance["current_geometry_node"]->set(mGeometryGroup);
	mGeometryInstance["mesh_id"]->setInt(mMeshID);
    initialize_buffer<optix::Aabb>(mBBoxBuffer, mMeshData.mBoundingBox);
    BufPtr<optix::Aabb> bptr = BufPtr<optix::Aabb>(mBBoxBuffer->getId());
    mGeometryInstance["local_bounding_box"]->setUserData(sizeof(BufPtr<optix::Aabb>), &bptr);
    mGeometry->markDirty();
	mReloadGeometry = false;
}

void Mesh::load_shader()
{
	if (mReloadShader)
	{
		mShader->initialize_mesh(*this);
		mReloadShader = false;
	}
	mShader->load_data();
}

void Mesh::load_transform()
{
	if (!mTransform)
	{
		mTransform = std::make_unique<Transform>();
	}
	if (mTransform->has_changed())
	{
		mOptixTransform->setMatrix(false, mTransform->get_matrix().getData(), nullptr);
		mGeometryGroup->getAcceleration()->markDirty();
		if(transform_changed_event != nullptr)
			transform_changed_event();
	}
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
	load_transform();
}

void Mesh::set_method(RenderingMethodType::EnumType method)
{
    mShader->set_method(method);
	mReloadShader = true;
	mReloadMaterials = true;
}

void Mesh::set_shader(int illum)
{
    mShader = ShaderFactory::get_shader(illum);
	mReloadShader = true;
	mReloadMaterials = true;
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

	if (!mOptixTransform)
	{
		mOptixTransform = mContext->createTransform();
		bind = true;
	}

	if (!mGeometryGroup)
	{
		mGeometryGroup = mContext->createGeometryGroup();
		bind = true;
	}

    if (bind)
    {
		mGeometryGroup->setChildCount(1);
        mGeometryInstance->setGeometry(mGeometry);
        mGeometryInstance->setMaterialCount(1);
        mGeometryInstance->setMaterial(0, mMaterial);

		optix::Acceleration acceleration = mContext->createAcceleration(std::string("Trbvh"));
		acceleration->setProperty("refit", "0");
		acceleration->setProperty("vertex_buffer_name", "vertex_buffer");
		acceleration->setProperty("index_buffer_name", "vindex_buffer");
		
		mGeometryGroup->setAcceleration(acceleration);
		acceleration->markDirty();
		mGeometryGroup->setChild(0, mGeometryInstance);

		mOptixTransform->setChild(mGeometryGroup);
    }    
}

void Mesh::add_material(std::shared_ptr<MaterialHost> material)
{
    mMaterialData.push_back(material);
    mMaterialBuffer->setSize(mMaterialData.size());
    load_materials();
}

bool Mesh::on_draw()
{
	bool changed = false;
	if (ImmediateGUIDraw::TreeNode((mMeshName + " ID: " + std::to_string(mMeshID)).c_str()))
	{
		if (ImmediateGUIDraw::TreeNode((std::string("Transform##Transform") + mMeshName).c_str()))
		{
			changed |= mTransform->on_draw();
			ImmediateGUIDraw::TreePop();
		}
		
		auto map = ShaderFactory::get_map();
		std::vector<std::string> vv;
		std::vector<int> illummap;
		int initial = mShader->get_illum();
		int starting = 0;
		for (auto& p : map)
		{
			vv.push_back(p.second->get_name());
			illummap.push_back(p.first);
			if (p.first == initial)
				starting = (int)illummap.size()-1;
		}
		std::vector<const char *> v;
		for (auto& c : vv) v.push_back(c.c_str());

		int selected = starting;
		if (ImmediateGUIDraw::TreeNode((std::string("Shader##Shader") + mMeshName).c_str()))
		{
			if (ImGui::Combo((std::string("Set Shader##RenderingMethod") + mMeshName).c_str(), &selected, v.data(), (int)v.size(), (int)v.size()))
			{
				changed = true;
				set_shader(illummap[selected]);
			}

			changed |= mShader->on_draw();
			ImmediateGUIDraw::TreePop();
		}

		
		if (ImmediateGUIDraw::TreeNode((std::string("Materials##Materials") + mMeshName).c_str()))
		{
			for (auto& m : mMaterialData)
			{
				changed |= m->on_draw(mMeshName);
			}
			ImmediateGUIDraw::TreePop();
		}

		ImmediateGUIDraw::TreePop();
	}
	return changed;
}

void Mesh::pre_trace()
{
	mShader->pre_trace_mesh(*this);
}

void Mesh::post_trace()
{
	mShader->post_trace_mesh(*this);
}
