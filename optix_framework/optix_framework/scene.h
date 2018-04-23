//
// Created by alcor on 4/23/18.
//
#pragma once
#include <memory>
#include <vector>
#include <host_device_common.h>

class RenderingMethod;
class Object;
class Camera;

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

    std::string serialize();
    void set_method(std::unique_ptr<RenderingMethod> method);
    const RenderingMethod& get_method() const { return *method; }

    int add_object(std::unique_ptr<Object> object);
    int add_camera(std::unique_ptr<Camera> camera);

    void set_current_camera(int camera_id);
    void set_current_camera(std::unique_ptr<Camera> camera);
    std::shared_ptr<Camera> get_current_camera();
    std::shared_ptr<Camera> get_camera(int camera_id);

private:
    std::vector<std::shared_ptr<Object>> mMeshes;
    std::vector<std::shared_ptr<Camera>> mCameras;
    optix::Group scene;
    optix::Context context;
    std::unique_ptr<RenderingMethod> method;
    std::shared_ptr<Camera> mCurrentCamera;

    void transform_changed();
};

