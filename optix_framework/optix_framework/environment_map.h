#pragma once
#include <optix_world.h>
#include "host_device_common.h"

struct EnvmapProperties
{
    unsigned int importance_sample_envmap			DEFAULT(0);
    TexPtr environment_map_tex_id					DEFAULT(-1);
    optix::Matrix3x3 lightmap_rotation_matrix		DEFAULT(optix::Matrix3x3::identity());
    optix::float3 lightmap_multiplier				DEFAULT(optix::make_float3(1));
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

