#pragma once
#include <optix_world.h>
#include "host_device_common.h"

struct EnvmapProperties
{
    //__device__ EnvmapProperties() : importance_sample_envmap(0), environment_map_tex_id(-1), lightmap_rotation_matrix(optix::Matrix3x3::identity()), lightmap_multiplier(optix::make_float3(1)){}
    unsigned int importance_sample_envmap;
    TexPtr environment_map_tex_id;
    optix::Matrix3x3 lightmap_rotation_matrix;
    optix::float3 lightmap_multiplier;
};

struct EnvmapImportanceSamplingData
{
    rtBufferId<float> marginal_pdf;
    rtBufferId<float, 2> conditional_pdf;
    rtBufferId<float> marginal_cdf;
    rtBufferId<float, 2> conditional_cdf;
    rtBufferId<float, 2> env_luminance;
    rtBufferId<float> marginal_f;
};

