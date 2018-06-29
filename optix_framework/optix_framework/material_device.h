#pragma once
#include <device_common.h>
#include <material_common.h>
#include <mesh_device.h>

rtDeclareVariable(MaterialDataCommon, main_material, , );
rtDeclareVariable(TexPtr, material_selector, , );

rtBuffer<MaterialDataCommon, 1> material_buffer;

_fn unsigned int get_material_index(const optix::float2 & uv)
{
    return optix::rtTex2D<int>(material_selector, uv.x, uv.y);
}

_fn MaterialDataCommon get_material(const optix::float2 & uv)
{
    unsigned int idx = get_material_index(uv);
    optix_assert(idx < material_buffer.size());
    if (idx >= material_buffer.size())
        idx = 0;
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

_fn float get_monochromatic_ior(const MaterialDataCommon& mat)
{
    return dot(mat.index_of_refraction, optix::make_float3(1.0f / 3.0f));
}

_fn float get_monochromatic_roughness(const MaterialDataCommon& mat)
{
    return dot(mat.roughness, optix::make_float2(0.5f));
}