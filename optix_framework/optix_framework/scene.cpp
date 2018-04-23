//
// Created by alcor on 4/23/18.
//

#include "scene.h"
#include <functional>
#include "object_host.h"
#include "rendering_method.h"
#include "camera_host.h"
#include <algorithm>
#include "scattering_material.h"

void Scene::execute_on_scene_elements(std::function<void(Object&)> operation)
{
    for (auto & m : mMeshes)
    {
        operation(*m);
    }
}

void Scene::reload()
{
    execute_on_scene_elements([=](Object & m)
    {
        m.reload_shader();
    });
}

Scene::Scene(optix::Context context)
{
    this->context = context;
    mMeshes.clear();
    scene = context->createGroup();
    optix::Acceleration acceleration = context->createAcceleration("Bvh");
    scene->setAcceleration(acceleration);
    acceleration->markDirty();
    scene->setChildCount(0);
    context["top_object"]->set(scene);
    context["top_shadower"]->set(scene);
}

bool Scene::on_draw()
{
    bool changed = false;
    execute_on_scene_elements([&](Object & m)
    {
        changed |= m.on_draw();
    });

    changed |= mCurrentCamera->on_draw();
    return changed;
}

std::string Scene::serialize()
{
    std::stringstream ss;
    {
        cereal::XMLOutputArchive output_archive(ss);
        output_archive(mMeshes[0]);
    }
    Logger::info << ss.str();

    //cereal::XMLInputArchiveOptix iarchive(context, ss);
    //std::unique_ptr<Object> m;
    //iarchive(m);
}

void Scene::pre_trace()
{
    mCurrentCamera->set_into_gpu(context);
    method->pre_trace();
    execute_on_scene_elements([=](Object & m)
    {
        m.load();
        m.pre_trace();
    });
}

void Scene::post_trace()
{
    method->post_trace();
    execute_on_scene_elements([=](Object & m)
    {
        m.post_trace();
    });
}

void Scene::trace()
{
    context->launch(mCurrentCamera->get_entry_point(), mCurrentCamera->get_width(), mCurrentCamera->get_height());
}

void Scene::set_method(std::unique_ptr<RenderingMethod> m)
{
    method = std::move(m);
    method->init();
    execute_on_scene_elements([=](Object & m)
    {
        m.reload_shader();
        m.reload_material();
    });
}

int Scene::add_object(std::unique_ptr<Object> object)
{
    object->transform_changed_event = std::bind(&Scene::transform_changed, this);
    object->scene = this;
    mMeshes.push_back(std::shared_ptr<Object>(std::move(object)));
    scene->addChild(mMeshes.back()->get_dynamic_handle());
    mMeshes.back()->load();
    return mMeshes.size();
}

void Scene::transform_changed()
{
    scene->getAcceleration()->markDirty();
}

int Scene::add_camera(std::unique_ptr<Camera> camera)
{
    mCameras.push_back(std::move(camera));
    return mCameras.back()->get_id();
}

void Scene::set_current_camera(int camera_id)
{
    auto val = std::find_if(mCameras.begin(), mCameras.end(), [=](std::shared_ptr<Camera>&camera){ return camera->get_id() == camera_id; });
    if(val != mCameras.end())
    {
        mCurrentCamera = *val;
    }
}

void Scene::set_current_camera(std::unique_ptr<Camera> camera)
{
    int camera_id = camera->get_id();
    auto val = std::find_if(mCameras.begin(), mCameras.end(), [=](std::shared_ptr<Camera>&cam){ return camera_id == cam->get_id(); });
    if(val != mCameras.end())
    {
        mCurrentCamera = *val;
    }
    else
    {
        add_camera(std::move(camera));
        mCurrentCamera = mCameras.back();
    }

}

std::shared_ptr<Camera> Scene::get_current_camera()
{
    return mCurrentCamera;
}

std::shared_ptr<Camera> Scene::get_camera(int camera_id)
{
    auto val = std::find_if(mCameras.begin(), mCameras.end(), [=](std::shared_ptr<Camera>&camera){ return camera->get_id() == camera_id; });
    if(val != mCameras.end())
    {
        return *val;
    }
}

