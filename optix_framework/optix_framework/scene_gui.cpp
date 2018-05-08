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

std::shared_ptr<MaterialHost> get_default_material(optix::Context ctx)
{
    ObjMaterial objmat;

    objmat.ambient_tex = std::move(createOneElementSampler(ctx, optix::make_float4(0)));
    objmat.diffuse_tex = std::move(createOneElementSampler(ctx, optix::make_float4(1)));
    objmat.specular_tex = std::move(createOneElementSampler(ctx, optix::make_float4(0)));

    auto b = new MaterialHost(ctx, objmat);
    return std::shared_ptr<MaterialHost>(b);
}

bool SceneGUI::on_draw(Scene * scene)
{
    bool changed = false;

    optix::Context context = scene->context;

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

        static int primitive = 0;
        ImmediateGUIDraw::Combo("Primitive", &primitive, "Sphere\0Plint\0Box", 3);

        if(ImmediateGUIDraw::Button("Add primitive"))
        {
            std::shared_ptr<Object> c = std::make_shared<Object>(context);
            std::unique_ptr<Geometry> g;
            std::string name;
            if(primitive == 0)
            {
                g = std::make_unique<Sphere>(context, optix::make_float3(0), 1.0f);
                name = "sphere";
            }
            else if (primitive == 1){
                g = std::make_unique<Plint>(context, optix::make_float3(0), 1, 2, 1);
                name = "plint";
            }
            else
            {
                g = std::make_unique<Box>(context, optix::make_float3(0), 1, 1, 1);
                name = "box";

            }
            auto m = get_default_material(context);
            c->init(name.c_str(), std::move(g), m);
            scene->add_object(c);
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
