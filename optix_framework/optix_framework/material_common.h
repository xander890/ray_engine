#pragma once
#include "host_device_common.h"
#include "scattering_properties.h"

struct MaterialDataCommon
{
    int    illum;
	optix::float3  index_of_refraction;
	float  roughness;
	float  anisotropy_angle;
    TexPtr ambient_map;
    TexPtr diffuse_map;
    TexPtr specular_map;
    ScatteringMaterialProperties scattering_properties;
	int test;
};