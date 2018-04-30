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
    mLights.clear();
    mCameras.clear();

    mAreaLightBuffer = create_buffer<TriangleLight>(context, RT_BUFFER_INPUT, 0);
    mSingularLightBuffer = create_buffer<SingularLightData>(context, RT_BUFFER_INPUT, 0);
}

bool Scene::on_draw()
{
    bool changed = false;

    if (ImmediateGUIDraw::CollapsingHeader("Meshes"))
    {
        execute_on_scene_elements([&](Object &m)
        {
            changed |= m.on_draw();
        });
    }

    if (ImmediateGUIDraw::CollapsingHeader("Camera"))
    {
        changed |= mCurrentCamera->on_draw();
    }

    if (ImmediateGUIDraw::CollapsingHeader("Background"))
    {
        changed |= miss_program->on_draw();
    }

    if (ImmediateGUIDraw::CollapsingHeader("Lights"))
    {
        if(ImmediateGUIDraw::Button("Add light"))
        {
            changed = true;
            std::unique_ptr<SingularLight> l = std::make_unique<SingularLight>();
            add_light(std::move(l));
        }
        for(auto & l : mLights)
        {
            changed |= l->on_draw();
        }
    }

    return changed;
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
    update_area_lights();
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

void Scene::set_miss_program(std::unique_ptr<MissProgram> miss)
{
    miss->init(context);
    miss_program = std::move(miss);
}

int Scene::add_light(std::unique_ptr<SingularLight> light)
{
    light->init(context);
    mLights.push_back(std::move(light));
    mSingularLightBuffer->setSize(mLights.size());
    update_singular_lights();
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
    context["area_lights"]->setBuffer(mAreaLightBuffer);
}

void Scene::update_singular_lights()
{
    SingularLightData * l = reinterpret_cast<SingularLightData*>(mSingularLightBuffer->map());
    for(int i = 0; i < mLights.size(); i++)
    {
        l[i] = mLights[i]->get_data();
    }
    mSingularLightBuffer->unmap();
    context["singular_lights"]->setBuffer(mSingularLightBuffer);
}

