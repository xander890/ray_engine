//
// Created by alcor on 5/1/18.
//

#include "scene_gui.h"
#include "scene.h"
#include "immediate_gui.h"
#include "object_host.h"
#include "camera_host.h"
#include <algorithm>
#include <fstream>
#include "miss_program.h"
#include "light_host.h"
#include "dialogs.h"
#include "obj_loader.h"
#include "sphere.h"
#include "ImageLoader.h"
#include "plint.h"
#include "box.h"
#include "environment_map_background.h"
#include "sky_model.h"
#include "constant_background.h"


bool SceneGUI::on_draw(Scene * scene)
{
    bool changed = false;

    optix::Context context = scene->context;

    if (ImmediateGUIDraw::CollapsingHeader("Meshes"))
    {
        if(ImmediateGUIDraw::Button("Load from file..."))
        {
            std::string path;
            if(Dialogs::openFileDialog(path))
            {
                ObjLoader* loader = new ObjLoader(path.c_str(), scene->context);
                std::vector<std::unique_ptr<Object>>& v = loader->load();
                for (auto& c : v)
                {
                    scene->add_object(std::move(c));
                }
                changed = true;
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Add primitive..."))
            ImGui::OpenPopup("addprimitivepopup");
        ImGui::SameLine();

        if (ImGui::Button("Remove mesh..."))
            ImGui::OpenPopup("removemeshpopup");

        if (ImGui::BeginPopup("removemeshpopup"))
        {
            for (int i = 0; i < scene->mMeshes.size(); i++)
            {
                if (ImGui::Selectable(scene->mMeshes[i]->get_name().c_str()))
                {
                    scene->remove_object(i);
                }
            }
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("addprimitivepopup"))
        {

            if (ImGui::Selectable("Sphere"))
            {
                std::shared_ptr<Object> c = std::make_shared<Object>(context);
                auto g = std::make_unique<Sphere>(context, optix::make_float3(0), 1.0f);
                std::string name = "sphere";
                auto m = get_default_material(context);
                c->init(name.c_str(), std::move(g), m);
                scene->add_object(c);
            }
            if (ImGui::Selectable("Plint"))
            {
                std::shared_ptr<Object> c = std::make_shared<Object>(context);
                auto g = std::make_unique<Plint>(context, optix::make_float3(0), 1, 2, 1);
                std::string name = "plint";
                auto m = get_default_material(context);
                c->init(name.c_str(), std::move(g), m);
                scene->add_object(c);
            }
            if (ImGui::Selectable("Box"))
            {
                std::shared_ptr<Object> c = std::make_shared<Object>(context);
                auto g = std::make_unique<Box>(context, optix::make_float3(0), 1, 1, 1);
                std::string name = "box";
                auto m = get_default_material(context);
                c->init(name.c_str(), std::move(g), m);
                scene->add_object(c);
            }

            ImGui::EndPopup();
        }

        if(ImmediateGUIDraw::TreeNode("Meshes"))
        {
            scene->execute_on_scene_elements([&](Object &m)
            {
                changed |= m.on_draw();
            });
            ImmediateGUIDraw::TreePop();
        }
    }

    if (ImmediateGUIDraw::CollapsingHeader("Camera"))
    {
        changed |= scene->get_current_camera()->on_draw();
    }

    if (ImmediateGUIDraw::CollapsingHeader("Background"))
    {

        if (ImGui::Button("Change background..."))
            ImGui::OpenPopup("changebackgroundpopup");

        if (ImGui::BeginPopup("changebackgroundpopup"))
        {
            if (ImGui::Selectable("Constant"))
            {
                scene->set_miss_program(std::make_unique<ConstantBackground>(optix::make_float3(0.5f)));
            }

            if (ImGui::Selectable("Environment map"))
            {
                std::string path;
                if(Dialogs::openFileDialog(path))
                {
                    scene->set_miss_program(std::make_unique<EnvironmentMap>(path));
                }
            }

            if (ImGui::Selectable("Sky model"))
            {
                scene->set_miss_program(std::make_unique<SkyModel>(optix::make_float3(0, 1, 0), optix::make_float3(0, 0, 1)));
            }


            ImGui::EndPopup();
        }

        changed |= scene->miss_program->on_draw();
    }

    if (ImmediateGUIDraw::CollapsingHeader("Lights"))
    {
        if(ImmediateGUIDraw::Button("Add light"))
        {
            changed = true;
            std::unique_ptr<SingularLight> l = std::make_unique<SingularLight>();
            scene->add_light(std::move(l));
        }
        for(auto & l : scene->mLights)
        {
            changed |= l->on_draw();
        }
    }

    return changed;}
