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
#include "optix_serialize.h"
#include "optix_utils.h"
#include "mesh.h"

class Scene;

class Object
{
public:
    friend class Scene;
    explicit Object(optix::Context ctx);

    void init(const char* name, std::unique_ptr<Geometry> geom, std::shared_ptr<MaterialHost> material);

    std::shared_ptr<Geometry> mGeometry;
    optix::Context  mContext;
    optix::Material mMaterial = nullptr;

    void reload_shader();
    void reload_material();
    void load();

    void set_shader(int illum);
    void set_shader(const std::string & source);
    void add_material(std::shared_ptr<MaterialHost> material);

    std::shared_ptr<MaterialHost> get_main_material() { return mMaterialData[0]; }
    const std::vector<std::shared_ptr<MaterialHost>> & get_materials() { return mMaterialData; }

    bool on_draw();
    void pre_trace();
    void post_trace();

    optix::GeometryInstance get_geometry_instance() { return mGeometryInstance;  }
    optix::GeometryGroup get_static_handle() { return mGeometryGroup; }
    optix::Transform get_dynamic_handle() { return mTransform->get_transform(); }

    typedef std::function<void()> TransformChangedDelegate;
    TransformChangedDelegate transform_changed_event = nullptr;

    const Scene& get_scene() const { return *scene; }


private:
    Object() {}
    optix::GeometryInstance mGeometryInstance = nullptr;
    optix::GeometryGroup mGeometryGroup = nullptr;

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
    void load(Archive & archive)
    {
        archive(cereal::make_nvp("name", mMeshName));
        //archive(cereal::make_nvp("data", mMeshData));
        //archive(cereal::make_nvp("materials",mMaterialData));
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("name", mMeshName));
        //archive(cereal::make_nvp("data", mMeshData));
        archive(cereal::make_nvp("materials",mMaterialData));
    }

    int mMeshID;
    Scene* scene;
    friend class Scene;
};