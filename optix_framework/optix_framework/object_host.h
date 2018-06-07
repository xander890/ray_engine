#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"
#include <memory>
#include "rendering_method.h"
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
    ~Object();

    void init(const char* name, std::unique_ptr<Geometry> geom, std::shared_ptr<MaterialHost> material);

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

    std::shared_ptr<Geometry> get_geometry() { return mGeometry; }

    optix::GeometryInstance get_geometry_instance() { return mGeometryInstance;  }
    optix::GeometryGroup get_static_handle() { return mGeometryGroup; }
    optix::Transform get_dynamic_handle() { return mTransform->get_transform(); }

    typedef std::function<void()> TransformChangedDelegate;
    TransformChangedDelegate transform_changed_event = nullptr;

    const Scene& get_scene() const { return *scene; }
    const std::string& get_name() { return mMeshName; }

    optix::Material mMaterial = nullptr;

private:

    void load_materials();
    void load_geometry();
    void load_shader();
    void load_transform();
    void create_and_bind_optix_data();

    friend class cereal::access;
    // Serialization
    static void load_and_construct( cereal::XMLInputArchiveOptix & archive, cereal::construct<Object> & construct )
    {
        construct(archive.get_context());
        archive(cereal::make_nvp("name", construct->mMeshName));
        archive(cereal::make_nvp("geometry", construct->mGeometry));
        archive(cereal::make_nvp("transform", construct->mTransform));
        archive(cereal::make_nvp("materials",construct->mMaterialData));
        archive(cereal::make_nvp("shader",construct->mShader));
        construct->create_and_bind_optix_data();
        construct->mReloadMaterials = construct->mReloadShader = construct->mReloadGeometry = true;
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("name", mMeshName));
        archive(cereal::make_nvp("geometry", mGeometry));
        archive(cereal::make_nvp("transform", mTransform));
        archive(cereal::make_nvp("materials",mMaterialData));
        archive(cereal::make_nvp("shader", mShader));
    }

    int mMeshID;
    Scene* scene;
    friend class Scene;

    optix::GeometryInstance mGeometryInstance = nullptr;
    optix::GeometryGroup mGeometryGroup = nullptr;
    optix::Acceleration mAcceleration = nullptr;

    std::shared_ptr<Geometry> mGeometry;
    optix::Context  mContext;
    std::unique_ptr<Shader> mShader;
    std::unique_ptr<Transform> mTransform;
    std::vector<std::shared_ptr<MaterialHost>> mMaterialData;
    optix::Buffer          mMaterialBuffer;
    std::string            mMeshName;
    std::unique_ptr<Texture> mMaterialSelectionTextureLabel;
    std::unique_ptr<Texture> mMaterialSelectionTexture;


    bool mReloadShader = true;
    bool mReloadGeometry = true;
    bool mReloadMaterials = true;

};