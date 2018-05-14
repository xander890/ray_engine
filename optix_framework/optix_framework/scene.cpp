//
// Created by alcor on 4/23/18.
//

#include "scene.h"
#include <functional>
#include "object_host.h"
#include "rendering_method.h"
#include "camera_host.h"
#include <algorithm>
#include <fstream>
#include "scattering_material.h"
#include "miss_program.h"
#include "immediate_gui.h"
#include "light_host.h"
#include "area_light.h"
#include "texture.h"
#include "scene_gui.h"

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
    acceleration = context->createAcceleration("Bvh");
    scene->setAcceleration(acceleration);
    acceleration->markDirty();
    scene->setChildCount(0);
    context["top_object"]->set(scene);
    context["top_shadower"]->set(scene);
    mLights.clear();
    mCameras.clear();

    mAreaLightBuffer = create_buffer<TriangleLight>(context, RT_BUFFER_INPUT, 0);
    mSingularLightBuffer = create_buffer<SingularLightData>(context, RT_BUFFER_INPUT, 0);

    context["singular_lights"]->setBuffer(mSingularLightBuffer);
    context["area_lights"]->setBuffer(mAreaLightBuffer);

    mGUI = std::make_unique<SceneGUI>();

    CameraParameters def;
    mCurrentCamera = std::make_shared<Camera>(context, def);
}

bool Scene::on_draw()
{
    return mGUI->on_draw(this);
}


void Scene::pre_trace()
{
    miss_program->set_into_gpu(context);
    method->pre_trace();

    update_singular_lights();

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
    method->init(context);
    execute_on_scene_elements([=](Object & m)
    {
        m.reload_shader();
        m.reload_material();
    });
}

int Scene::add_object(std::shared_ptr<Object> object)
{
    object->transform_changed_event = std::bind(&Scene::transform_changed, this);
    object->scene = this;
    mMeshes.push_back(object);
    scene->addChild(mMeshes.back()->get_dynamic_handle());
    mMeshes.back()->load();
    update_area_lights();
    return mMeshes.size();
}

int Scene::add_camera(std::shared_ptr<Camera> camera)
{
    mCameras.push_back(camera);
    return mCameras.back()->get_id();
}

int Scene::add_light(std::shared_ptr<SingularLight> light)
{
    light->init(context);
    mLights.push_back(light);
    mSingularLightBuffer->setSize(mLights.size());
    update_singular_lights();
	return mLights.size() - 1;
}

int Scene::add_object(std::unique_ptr<Object> object)
{
    return add_object(std::shared_ptr<Object>(std::move(object)));
}

int Scene::add_camera(std::unique_ptr<Camera> camera)
{
    return add_camera(std::shared_ptr<Camera>(std::move(camera)));
}

int Scene::add_light(std::unique_ptr<SingularLight> light)
{
    return add_light(std::shared_ptr<SingularLight>(std::move(light)));
}

void Scene::transform_changed()
{
    scene->getAcceleration()->markDirty();
}



void Scene::set_current_camera(int camera_id)
{
    auto val = std::find_if(mCameras.begin(), mCameras.end(), [=](std::shared_ptr<Camera>&camera){ return camera->get_id() == camera_id; });
    if(val != mCameras.end())
    {
        mCurrentCamera->setAsOtherCamera(*val);
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
	return nullptr;
}

void Scene::set_miss_program(std::unique_ptr<MissProgram> miss)
{
    miss->init(context);
    miss_program = std::move(miss);
}

void Scene::update_area_lights()
{
    std::vector<TriangleLight> lights;
    execute_on_scene_elements([&](Object & obj)
    {
        if(!obj.get_main_material()->is_emissive())
        {
            return;
        }
        auto texture = obj.get_main_material()->get_ambient_texture();
        optix::float3 color = optix::make_float3(texture->get_texel(0,0,0));

        std::vector<optix::float3> vs;
        obj.mGeometry->get_flattened_vertices(vs);

        for(int i = 0; i < vs.size() / 3; i++)
        {
            TriangleLight light;
            light.v1 = vs[i * 3 + 0];
            light.v2 = vs[i * 3 + 1];
            light.v3 = vs[i * 3 + 2];

            float3 area_vec = cross(light.v2 - light.v1, light.v3 - light.v1);
            light.area = 0.5f* length(area_vec);
            // normal vector
            light.normal = normalize(area_vec);
            light.emission = color;
            lights.push_back(light);
        }
    });

    if(mAreaLightBuffer.get() == nullptr)
    {
        mAreaLightBuffer = context->createBuffer(RT_BUFFER_INPUT);
        mAreaLightBuffer->setFormat(RT_FORMAT_USER);
        mAreaLightBuffer->setElementSize(sizeof(TriangleLight));
    }
    mAreaLightBuffer->setSize(lights.size());
    void * buf = mAreaLightBuffer->map();
    memcpy(buf, &lights[0], lights.size() * sizeof(TriangleLight));
    mAreaLightBuffer->unmap();
}

void Scene::update_singular_lights()
{
    SingularLightData * l = reinterpret_cast<SingularLightData*>(mSingularLightBuffer->map());
    for(int i = 0; i < mLights.size(); i++)
    {
        l[i] = mLights[i]->get_data();
    }
    mSingularLightBuffer->unmap();
}

void Scene::remove_object(int object_id)
{
    if(object_id >= 0 && object_id < mMeshes.size())
    {
        scene->removeChild(mMeshes[object_id]->get_dynamic_handle());
        mMeshes.erase(mMeshes.begin() + object_id);
        scene->getAcceleration()->markDirty();
        for(auto & m : mMeshes)
        {
            m->mAcceleration->markDirty();
            m->mGeometry->get_geometry()->markDirty();
        }
        update_area_lights();
    }
}

void Scene::remove_camera(int camera_id)
{
    if(camera_id >= 0 && camera_id < mCameras.size())
    {
        mCameras.erase(mCameras.begin() + camera_id);
    }
}

void Scene::remove_light(int light_id)
{
    if(light_id >= 0 && light_id < mLights.size())
    {
        mLights.erase(mLights.begin() + light_id);
        update_singular_lights();
    }
}

Scene::~Scene()
{
    mLights.clear();
    mCameras.clear();
    mMeshes.clear();
    scene->destroy();
    acceleration->destroy();
    method.reset();
    miss_program.reset();
    mAreaLightBuffer->destroy();
    mSingularLightBuffer->destroy();
}

