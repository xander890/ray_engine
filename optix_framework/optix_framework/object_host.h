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

    void init(const char* name, std::unique_ptr<Geometry> geom, std::shared_ptr<MaterialHost>& material);

    void reload_materials();
    void load();

    void add_material(std::shared_ptr<MaterialHost> material);

    std::shared_ptr<MaterialHost> get_main_material() { return mMaterials[0]; }
    const std::vector<std::shared_ptr<MaterialHost>> & get_materials() { return mMaterials; }

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

    optix::GeometryInstance& get_instance() {return mGeometryInstance; }

private:

    void load_materials();
    void load_geometry();
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
        archive(cereal::make_nvp("materials",construct->mMaterials));
        archive(cereal::make_nvp("material_selector", construct->mMaterialSelectionTexture));

        construct->create_and_bind_optix_data();
        construct->mReloadMaterials = construct->mReloadGeometry = true;
        construct->mMaterialSelectionTextureLabel = create_label_texture(archive.get_context(), construct->mMaterialSelectionTexture);
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("name", mMeshName));
        archive(cereal::make_nvp("geometry", mGeometry));
        archive(cereal::make_nvp("transform", mTransform));
        archive(cereal::make_nvp("materials", mMaterials));
        archive(cereal::make_nvp("material_selector", mMaterialSelectionTexture));
    }

    int mMeshID;
    Scene* scene;
    friend class Scene;

    optix::GeometryInstance mGeometryInstance = nullptr;
    optix::GeometryGroup mGeometryGroup = nullptr;
    optix::Acceleration mAcceleration = nullptr;

    std::shared_ptr<Geometry> mGeometry = nullptr;
    optix::Context  mContext = nullptr;
    std::unique_ptr<Transform> mTransform = nullptr;
    std::vector<std::shared_ptr<MaterialHost>> mMaterials;
    optix::Buffer          mMaterialBuffer = nullptr;
    std::string            mMeshName = "";
    std::unique_ptr<Texture> mMaterialSelectionTextureLabel = nullptr;
    std::unique_ptr<Texture> mMaterialSelectionTexture = nullptr;

    bool mReloadGeometry = true;
    bool mReloadMaterials = true;
    static std::unique_ptr<Texture> create_label_texture(optix::Context ctx, const std::unique_ptr<Texture>& ptr);
};