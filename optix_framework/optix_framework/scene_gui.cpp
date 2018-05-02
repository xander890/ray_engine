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

bool SceneGUI::on_draw(Scene * scene)
{
    bool changed = false;

    if (ImmediateGUIDraw::CollapsingHeader("Meshes"))
    {
        const char * mesh_names[100];
        for(int i = 0; i < scene->mMeshes.size(); i++)
        {
            mesh_names[i] = scene->mMeshes[i]->get_name().c_str();
        }

        static int deletion_req = 0;
        ImmediateGUIDraw::Combo("Selected mesh", &deletion_req, mesh_names, scene->mMeshes.size(), scene->mMeshes.size());

        if(ImmediateGUIDraw::Button("Load obj file..."))
        {
            std::string path;
            if(Dialogs::openFileDialog(path, "*.obj"))
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
        ImmediateGUIDraw::SameLine();
        if(ImmediateGUIDraw::Button("Remove mesh"))
        {
            scene->remove_object(deletion_req);
            changed = true;
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
        changed |= scene->mCurrentCamera->on_draw();
    }

    if (ImmediateGUIDraw::CollapsingHeader("Background"))
    {
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
