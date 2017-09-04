#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"
#include <memory>
#include "rendering_method.h"

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
    optix::Aabb   mBoundingBox;
};

class Mesh
{
public:
    explicit Mesh(optix::Context ctx);

    void init(const char* name, MeshData meshdata, std::shared_ptr<MaterialHost> material);

    optix::GeometryInstance mGeometryInstance = nullptr;
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

    void set_into_gui(GUI * gui, const char * group = "");
    void remove_from_gui(GUI * gui);
	void pre_trace();

private:
	MeshData mMeshData;
	std::shared_ptr<Shader> mShader;

	void load_materials();
	void load_geometry();
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

    static void GUI_CALL setShader(const void* var, void* data);
    static void GUI_CALL getShader(void* var, void* data);
};
