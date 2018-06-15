//
// Created by alcor on 4/23/18.
//
#pragma once
#include <memory>
#include <vector>
#include <host_device_common.h>
#include <optix_serialize.h>
#include "scene_gui.h"
#include "miss_program.h"
#include "light_host.h"
#include "camera_host.h"
#include "object_host.h"

class RenderingMethod;

class Scene
{
public:
    Scene(optix::Context context);
    ~Scene();

    void execute_on_scene_elements(std::function<void(Object&)> operation);

    void reload();
    bool on_draw();

    void pre_trace();
    void trace();
    void post_trace();

    void set_method(std::unique_ptr<RenderingMethod> method);
    const RenderingMethod& get_method() const { return *method; }
    void set_miss_program(std::unique_ptr<MissProgram> method);
    const MissProgram& get_miss_program() const { return *miss_program; }

    int add_object(std::unique_ptr<Object> object);
    int add_camera(std::unique_ptr<Camera> camera);
    int add_light(std::unique_ptr<SingularLight> light);
    int add_object(std::shared_ptr<Object> object);
    int add_camera(std::shared_ptr<Camera> camera);
    int add_light(std::shared_ptr<SingularLight> light);
    void remove_object(int object_id);
    void remove_camera(int camera_id);
    void remove_light(int light_id);


    void set_current_camera(int camera_id);
    std::shared_ptr<Camera> get_current_camera();
    std::shared_ptr<Camera> get_camera(int camera_id);

	optix::Acceleration get_acceleration() { return acceleration; }

private:
    std::vector<std::shared_ptr<Object>> mMeshes;
    std::vector<std::shared_ptr<Camera>> mCameras;
    std::vector<std::shared_ptr<SingularLight>> mLights;
    optix::Group scene;
    optix::Acceleration acceleration;
    optix::Context context;
    std::unique_ptr<RenderingMethod> method;
    std::unique_ptr<MissProgram> miss_program;
    int mCurrentCamera;

    void transform_changed();
    void update_area_lights();
    void update_singular_lights();
    optix::Buffer mAreaLightBuffer = nullptr;
    optix::Buffer mSingularLightBuffer = nullptr;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::make_nvp("miss_program", miss_program));
        archive(cereal::make_nvp("method", method));
        archive(cereal::make_nvp("lights", mLights));
        archive(cereal::make_nvp("cameras", mCameras));
        archive(cereal::make_nvp("current_camera", mCurrentCamera));
        archive(cereal::make_nvp("meshes", mMeshes));
    }

    static void load_and_construct( cereal::XMLInputArchiveOptix & ar, cereal::construct<Scene> & construct )
    {
        construct(ar.get_context());
        std::unique_ptr<MissProgram> miss_program;
        ar(cereal::make_nvp("miss_program", miss_program));
        construct->set_miss_program(std::move(miss_program));

        std::unique_ptr<RenderingMethod> method;
        ar(cereal::make_nvp("method", method));
        construct->set_method(std::move(method));

        std::vector<std::shared_ptr<SingularLight>> mLights;
        ar(cereal::make_nvp("lights", mLights));
        for(auto & l : mLights)
        {
            construct->add_light(l);
        }

        std::vector<std::shared_ptr<Camera>> mCameras;
        ar(cereal::make_nvp("cameras", mCameras));
        for(auto & c : mCameras)
        {
            construct->add_camera(c);
        }

        int c;
        ar(cereal::make_nvp("current_camera", c));

        std::vector<std::shared_ptr<Object>> mMeshes;
        ar(cereal::make_nvp("meshes", mMeshes));
        for(auto & m : mMeshes)
        {
            construct->add_object(m);
        }

        construct->set_current_camera(c);


        construct->update_singular_lights();
        construct->update_area_lights();
    }

    friend class SceneGUI;
    std::unique_ptr<SceneGUI> mGUI = nullptr;
};

