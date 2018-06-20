#pragma once

#include "material_host.h"
#include "obj_loader.h"
#include "image_loader.h"
#include <utility>

class Scene;

class SceneGUI
{
public:
    bool on_draw(Scene * scene);
};

inline std::shared_ptr<MaterialHost> get_default_material(optix::Context ctx)
{
    ObjMaterial objmat;

    objmat.ambient_tex = std::move(createOneElementSampler(ctx, optix::make_float4(0)));
    objmat.diffuse_tex = std::move(createOneElementSampler(ctx, optix::make_float4(1)));
    objmat.specular_tex = std::move(createOneElementSampler(ctx, optix::make_float4(0)));

    auto b = new MaterialHost(ctx, objmat);
    return std::shared_ptr<MaterialHost>(b);
}