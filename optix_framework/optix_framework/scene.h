//
// Created by alcor on 4/23/18.
//
#pragma once
#include <memory>
#include <vector>
#include <host_device_common.h>
#include <optix_serialize.h>
#include "scene_gui.h"

class RenderingMethod;
class Object;
class Camera;
class MissProgram;
class SingularLight;

class Scene
{
public:
    Scene(optix::Context context);

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
    void remove_object(int object_id);
    void remove_camera(int camera_id);
    void remove_light(int light_id);


    void set_current_camera(int camera_id);
    void set_current_camera(std::unique_ptr<Camera> camera);
    std::shared_ptr<Camera> get_current_camera();
    std::shared_ptr<Camera> get_camera(int camera_id);

private:
    std::vector<std::shared_ptr<Object>> mMeshes;
    std::vector<std::shared_ptr<Camera>> mCameras;
    std::vector<std::shared_ptr<SingularLight>> mLights;
    optix::Group scene;
    optix::Context context;
    std::unique_ptr<RenderingMethod> method;
    std::unique_ptr<MissProgram> miss_program;
    std::shared_ptr<Camera> mCurrentCamera;

    void transform_changed();
    void update_area_lights();
    void update_singular_lights();
    optix::Buffer mAreaLightBuffer = nullptr;
    optix::Buffer mSingularLightBuffer = nullptr;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(CEREAL_NVP(miss_program));
        archive(CEREAL_NVP(mLights));
        archive(CEREAL_NVP(mCameras));
        archive(CEREAL_NVP(mMeshes));
    }

    friend class SceneGUI;
    std::unique_ptr<SceneGUI> mGUI = nullptr;
};

