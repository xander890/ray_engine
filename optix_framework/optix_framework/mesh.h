#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"
#include <memory>
#include "rendering_method.h"
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include "cereal/types/memory.hpp"
#include "host_material.h"
#include "transform.h"

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
    optix::Aabb   mBoundingBox;
};

class Mesh
{
public:
    explicit Mesh(optix::Context ctx);

    void init(const char* name, MeshData meshdata, std::shared_ptr<MaterialHost> material);

    optix::Geometry mGeometry = nullptr;
    optix::Context  mContext;
    optix::Material mMaterial = nullptr;

	void reload_shader();
	void load();

    void set_method(RenderingMethodType::EnumType method);
    void set_shader(int illum);

    void add_material(std::shared_ptr<MaterialHost> material);

    std::shared_ptr<MaterialHost> get_main_material() { return mMaterialData[0]; }
    const std::vector<std::shared_ptr<MaterialHost>> & get_materials() { return mMaterialData; }

	bool on_draw();
	void pre_trace();
	void post_trace();

	optix::GeometryInstance get_geometry_instance() { return mGeometryInstance;  }
	optix::GeometryGroup get_static_handle() { return mGeometryGroup; }
	optix::Transform get_dynamic_handle() { return mOptixTransform; }

	typedef std::function<void()> TransformChangedDelegate;
	TransformChangedDelegate transform_changed_event = nullptr;

private:
	optix::GeometryInstance mGeometryInstance = nullptr;
	optix::GeometryGroup mGeometryGroup = nullptr;
	optix::Transform mOptixTransform = nullptr;

	MeshData mMeshData;
	std::unique_ptr<Shader> mShader;
	std::unique_ptr<Transform> mTransform;

	void load_materials();
	void load_geometry();
	void load_shader();
	void load_transform();

	std::vector<std::shared_ptr<MaterialHost>> mMaterialData;
    void create_and_bind_optix_data();
    optix::Program         mIntersectProgram;
    optix::Program         mBoundingboxProgram;
    optix::Buffer          mMaterialBuffer;
    optix::Buffer          mBBoxBuffer;
    std::string            mMeshName;

	bool mReloadShader = true;
	bool mReloadGeometry = true;
	bool mReloadMaterials = true;

	friend class cereal::access;
	// Serialization
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::make_nvp("name", mMeshName));
		archive(cereal::make_nvp("materials",mMaterialData));
	}
	int mMeshID;
};
