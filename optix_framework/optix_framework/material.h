#include "scattering_properties.h"

struct MaterialDataCommon
{
    optix::float3 emissive;
    optix::float3 reflectivity;
    optix::float3 absorption;
    float  phong_exp;
    float  ior;
    int    illum;
    int ambient_map;
    int diffuse_map;
    int specular_map;
    ScatteringMaterialProperties scattering_properties;
};