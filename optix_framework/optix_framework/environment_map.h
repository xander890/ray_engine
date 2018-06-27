#pragma once
#include <optix_world.h>
#include "host_device_common.h"

/*
 * Common data for environment map sampling
 */

/*
 * The actual Environment data. Includes a pointer to texture, options for importance sampling, a multiplier and a rotation.
 */
struct EnvmapProperties
{
    unsigned int importance_sample_envmap			DEFAULT(0);
    TexPtr environment_map_tex_id					DEFAULT(-1);
    optix::Matrix3x3 lightmap_rotation_matrix		DEFAULT(optix::Matrix3x3::identity());
    optix::float3 lightmap_multiplier				DEFAULT(optix::make_float3(1));
};

/*
 * Precomputed environment map sampling data (conditional and marginal distributions) to help with importance sampling.
 */
struct EnvmapImportanceSamplingData
{
    rtBufferId<float> marginal_pdf;
    rtBufferId<float, 2> conditional_pdf;
    rtBufferId<float> marginal_cdf;
    rtBufferId<float, 2> conditional_cdf;
    rtBufferId<float, 2> env_luminance;
    rtBufferId<float> marginal_f;
};

