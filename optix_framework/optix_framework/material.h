#pragma once
#include "host_device_common.h"
#include "scattering_properties.h"

struct MaterialDataCommon
{
    optix::float3 emissive;
    optix::float3 reflectivity;
    optix::float3 absorption;
    float  phong_exp;
    float  ior;
    int    illum;
    TexPtr ambient_map;
    TexPtr diffuse_map;
    TexPtr specular_map;
    ScatteringMaterialProperties scattering_properties;
    optix::float3 ior_complex_real_sq;
    optix::float3 ior_complex_imag_sq;
};