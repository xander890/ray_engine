#pragma once
#include <device_common_data.h>
#include <material.h>
#include <device_mesh_data.h>

using optix::rtTex3D;
using optix::float3;

rtDeclareVariable(MaterialDataCommon, main_material, , );
rtDeclareVariable(TexPtr, material_selector, , );

rtBuffer<MaterialDataCommon, 1> material_buffer;

__device__ __forceinline__ MaterialDataCommon get_material(const optix::float2 & uv)
{
    auto res = optix::rtTex2D<optix::int4>(material_selector, uv.x, uv.y);
    return material_buffer[res.x];
}

__device__ __forceinline__ MaterialDataCommon get_material(const optix::float3 & pos)
{
    return material_buffer[0];
}
