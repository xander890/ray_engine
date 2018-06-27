#pragma once
#include "device_common.h"
#include <optix_device.h>
#include <optix_math.h>
#include "random_device.h"
#include "merl_common.h"
#include "ray_tracing_utils.h"
#include "environment_map.h"
/*
 * Utility functions, mostly for importance sampling an environment map.
 */

using optix::float3;

rtDeclareVariable(int, envmap_enabled, , ) = 0;
rtDeclareVariable(EnvmapProperties, envmap_properties, , );
rtDeclareVariable(EnvmapImportanceSamplingData, envmap_importance_sampling, , );

_fn unsigned int cdf_bsearch_marginal(float xi)
{
    uint table_size = envmap_importance_sampling.marginal_cdf.size();
    uint middle = table_size = table_size >> 1;
    uint odd = 0;
    while (table_size > 0)
    {
        odd = table_size & 1;
        table_size = table_size >> 1;
        unsigned int tmp = table_size + odd;
        middle = xi > envmap_importance_sampling.marginal_cdf[middle]
            ? middle + tmp
            : (xi < envmap_importance_sampling.marginal_cdf[middle - 1] ? middle - tmp : middle);
    }
    return middle;
}

_fn unsigned int cdf_bsearch_conditional(float xi, uint offset)
{
    optix::size_t2 table_size = envmap_importance_sampling.conditional_cdf.size();
    uint middle = table_size.x = table_size.x >> 1;
    uint odd = 0;
    while (table_size.x > 0)
    {
        odd = table_size.x & 1;
        table_size.x = table_size.x >> 1;
        unsigned int tmp = table_size.x + odd;
        middle = xi > envmap_importance_sampling.conditional_cdf[optix::make_uint2(middle, offset)]
            ? middle + tmp
            : (xi < envmap_importance_sampling.conditional_cdf[optix::make_uint2(middle - 1, offset)] ? middle - tmp : middle);
    }
    return middle;
}

_fn void sample_environment(optix::float3& wi, optix::float3& radiance, const optix::float3& hit_point, const optix::float3& normal, TEASampler * sampler)
{
    if (envmap_enabled == 1)
    {
        optix::size_t2 count = envmap_importance_sampling.conditional_cdf.size();

        if (envmap_properties.importance_sample_envmap == 1 && count.x != 1 && count.y != 1)
        {
            auto marginal_cdf = envmap_importance_sampling.marginal_cdf;
            auto conditional_cdf = envmap_importance_sampling.conditional_cdf;
            auto marginal_pdf = envmap_importance_sampling.marginal_pdf;
            auto conditional_pdf = envmap_importance_sampling.conditional_pdf;

            float xi1 = sampler->next1D(), xi2 = sampler->next1D();

            uint v_idx = cdf_bsearch_marginal(xi1);
            float dv = v_idx > 0 && v_idx < count.y
                ? (xi1 - marginal_cdf[v_idx - 1]) / (marginal_cdf[v_idx] - marginal_cdf[v_idx - 1])
                : xi1 / marginal_cdf[v_idx];

            // Nasty bug solving, for when the cdf curve is flat
            dv = (v_idx > 0 && (marginal_cdf[v_idx] - marginal_cdf[v_idx - 1] < 10e-6)) ? 0 : dv;
            float pdf_m = marginal_pdf[v_idx];

            float v = (v_idx + dv) / count.y;
            uint u_idx = cdf_bsearch_conditional(xi2, v_idx);
            optix::uint2 uv_idx_prev = optix::make_uint2(u_idx - 1, v_idx);
            optix::uint2 uv_idx = optix::make_uint2(u_idx, v_idx);
            float du = u_idx > 0 && u_idx < count.x
                ? (xi2 - conditional_cdf[uv_idx_prev]) / (conditional_cdf[uv_idx] - conditional_cdf[uv_idx_prev])
                : xi2 / conditional_cdf[uv_idx];

            du = (u_idx > 0 && (conditional_cdf[uv_idx] - conditional_cdf[uv_idx_prev] < 10e-6)) ? 0 : du;
            float pdf_c = conditional_pdf[uv_idx];
            float u = (u_idx + du) / count.x;

            float probability = pdf_m*pdf_c;
            float theta = v*M_PIf;
            float phi = u*2.0f*M_PIf;
            float sin_theta, cos_theta, sin_phi, cos_phi;
            sincosf(theta, &sin_theta, &cos_theta);
            sincosf(phi, &sin_phi, &cos_phi);

            wi = optix::make_float3(sin_theta*sin_phi, -cos_theta, -sin_theta*cos_phi);
            float3 emission = optix::make_float3(0.0f);
            float V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX, emission);

            radiance = (probability < 1e-6) ? optix::make_float3(0.0f) : V * emission * sin_theta * 2.0f * M_PIf * M_PIf / probability;
            return;
        }
    }

	optix::float3 smp = sample_hemisphere_cosine(sampler->next2D(), normal);
    wi = normalize(smp);
    float3 emission = optix::make_float3(0.0f);
    float V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX, emission);
    radiance = emission * V * M_PIf;

}