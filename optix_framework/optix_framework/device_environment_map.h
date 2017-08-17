#ifndef ENVMAP_H
#define ENVMAP_H
#include "device_common_data.h"
#include <optix_device.h>
#include <optix_math.h>
#include "structs_device.h"
#include "random.h"
#include "merl_common.h"
#include "ray_trace_helpers.h"
#include "environment_map.h"
// Environment importance sampling
rtBuffer<float> marginal_pdf;
rtBuffer<float, 2> conditional_pdf;
rtBuffer<float> marginal_cdf;
rtBuffer<float, 2> conditional_cdf;

rtDeclareVariable(BufPtr<EnvmapProperties>, envmap_properties, , );

__forceinline__ __device__ unsigned int cdf_bsearch_marginal(float xi)
{
    uint table_size = marginal_cdf.size();
    uint middle = table_size = table_size >> 1;
    uint odd = 0;
    while (table_size > 0)
    {
        odd = table_size & 1;
        table_size = table_size >> 1;
        unsigned int tmp = table_size + odd;
        middle = xi > marginal_cdf[middle]
            ? middle + tmp
            : (xi < marginal_cdf[middle - 1] ? middle - tmp : middle);
    }
    return middle;
}

__forceinline__ __device__ unsigned int cdf_bsearch_conditional(float xi, uint offset)
{
    optix::size_t2 table_size = conditional_cdf.size();
    uint middle = table_size.x = table_size.x >> 1;
    uint odd = 0;
    while (table_size.x > 0)
    {
        odd = table_size.x & 1;
        table_size.x = table_size.x >> 1;
        unsigned int tmp = table_size.x + odd;
        middle = xi > conditional_cdf[make_uint2(middle, offset)]
            ? middle + tmp
            : (xi < conditional_cdf[make_uint2(middle - 1, offset)] ? middle - tmp : middle);
    }
    return middle;
}

__forceinline__ __device__ void sample_environment(optix::float3& wi, optix::float3& radiance, const HitInfo& data, unsigned int& seed)
{
    const optix::float3& hit_point = data.hit_point;
    const optix::float3& normal = data.hit_normal;
    optix::size_t2 count = conditional_cdf.size();

    if (envmap_properties->importance_sample_envmap == 1 && count.x != 1 && count.y != 1)
    {
        float xi1 = rnd(seed), xi2 = rnd(seed);

        uint v_idx = cdf_bsearch_marginal(xi1);
        float dv = v_idx > 0 && v_idx < count.y
            ? (xi1 - marginal_cdf[v_idx - 1]) / (marginal_cdf[v_idx] - marginal_cdf[v_idx - 1])
            : xi1 / marginal_cdf[v_idx];

        // Nasty bug solving, for when the cdf curve is flat
        dv = (v_idx > 0 && (marginal_cdf[v_idx] - marginal_cdf[v_idx - 1] < 10e-6)) ? 0 : dv;
        float pdf_m = marginal_pdf[v_idx];

        float v = (v_idx + dv) / count.y;
        uint u_idx = cdf_bsearch_conditional(xi2, v_idx);
        optix::uint2 uv_idx_prev = make_uint2(u_idx - 1, v_idx);
        optix::uint2 uv_idx = make_uint2(u_idx, v_idx);
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

        wi = make_float3(sin_theta*sin_phi, -cos_theta, -sin_theta*cos_phi);
        float3 emission = make_float3(0.0f);
        float V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX, emission);

        radiance = (probability < 1e-6) ? make_float3(0.0f) : V * emission * sin_theta * 2.0f * M_PIf * M_PIf / probability;
    }
    else
    {
        float zeta1 = rnd(seed);
        float zeta2 = rnd(seed);
        optix::float3 smp = sample_hemisphere_cosine(optix::make_float2(zeta1, zeta2), normal);
        wi = normalize(smp);
        float3 emission = optix::make_float3(0.0f);
        float V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX, emission);
        radiance = emission * V * M_PIf;
    }

}
#endif