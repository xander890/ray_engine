#pragma once
#include <device_common_data.h>
#include <material.h>
#include <device_mesh_data.h>

using optix::rtTex3D;
using optix::float3;

rtDeclareVariable(MaterialDataCommon, main_material, , );
rtDeclareVariable(TexPtr, material_selector, , );

rtBuffer<MaterialDataCommon, 1> material_buffer;

_fn unsigned int get_material_index(const optix::float2 & uv)
{
    auto res = optix::rtTex2D<optix::int4>(material_selector, uv.x, uv.y);
    return res.x;
}

_fn MaterialDataCommon get_material(const optix::float2 & uv)
{
    unsigned int idx = get_material_index(uv);
    optix_assert(idx < material_buffer.size());
    return material_buffer[idx];
}

_fn MaterialDataCommon get_material(const optix::float3 & pos)
{
    optix_assert(material_buffer.size() > 0);
    return material_buffer[0];
}

_fn MaterialDataCommon get_material()
{
    optix_assert(material_buffer.size() > 0);
    return material_buffer[0];
}
