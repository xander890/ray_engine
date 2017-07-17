#pragma once
#include "device_common_data.h"
#include <directional_light.h>
#include <point_light.h>
#include <area_light.h>
#include <ray_trace_helpers.h>
#include "host_device_common.h"

// Helper functions and data structures for lights. Note that every time a light is used this header must be included.

// Directional lights
rtBuffer<DirectionalLight, 1> directional_lights;
// Point light code
rtBuffer<PointLight, 1> point_lights;
// Area light code
rtBuffer<TriangleLight, 1> area_lights;

rtDeclareVariable(int, light_type, , );

__forceinline__ __device__
int no_light_size() { return 1; }

__forceinline__ __device__
void evaluate_no_light(const float3& test, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index) { radiance = make_float3(1, 0, 0); }

__forceinline__ __device__
int directional_light_size() { return directional_lights.size(); }

__forceinline__ __device__
void evaluate_directional_light(const float3 & hit_point, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index)
{
    DirectionalLight l = directional_lights[light_index];
    wi = -l.direction;

    float V = 1.0f;

    if (l.casts_shadow)
    {
        V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX);
    }

    radiance = V * l.emission;
    casts_shadows = l.casts_shadow;
}

__forceinline__ __device__
int point_light_size() { return point_lights.size(); }

__forceinline__ __device__
void evaluate_point_light(const float3 & hit_point, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index)
{
    PointLight l = point_lights[light_index];
    wi = l.position - hit_point;
    float dist_sq = dot(wi, wi);
    float dist = sqrt(dist_sq);
    wi /= dist;
    float V = 1.0f;
    if (l.casts_shadow)
    {
        V = trace_shadow_ray(hit_point, wi, scene_epsilon, dist);
    }
    radiance = l.intensity / dist_sq * max(dot(wi, hit_normal), 0.0f);
    casts_shadows = l.casts_shadow;
}

__forceinline__ __device__
int area_light_size() { return 1; } // This means that we will randomly sample from triangle lights instead of going though all of them.

__device__ __inline__ void evaluate_area_light_inline(const float3& hit_point, const float3& normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index, float tmin = scene_epsilon)
{
    //	assert(data != NULL);
    float zeta1 = rnd(seed);
    seed = lcg(seed);
    float zeta2 = rnd(seed);
    seed = lcg(seed);
    float zeta3 = rnd(seed);
    seed = lcg(seed);
    TriangleLight triangle = area_lights[(int)(area_lights.size()* zeta1)];
    optix::float3 point = sample_point_triangle(zeta2, zeta3, triangle.v1, triangle.v2, triangle.v3);
    optix::float3 to_light_un = point - hit_point;
    float dist_sq = dot(to_light_un, to_light_un);
    float dist = sqrt(dist_sq);
    wi = to_light_un / dist;
    float V = trace_shadow_ray(hit_point, wi, tmin, dist - scene_epsilon);
    casts_shadows = 1;
    radiance = V * triangle.emission * max(dot(wi, normal), 0.0f) * max(dot(-wi, triangle.normal), 0.0f) / dist_sq * triangle.area * area_lights.size();
}

__forceinline__ __device__
void evaluate_area_light(const float3 & hit_point, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index)
{
    //	assert(data != NULL);
    evaluate_area_light_inline(hit_point, hit_normal, wi, radiance, casts_shadows, seed, light_index, scene_epsilon);
}

/***
Cosine- hemisphere uniform sampling of the environment light map.
*/

__device__ __inline__ void evaluate_environment_light(optix::float3& wi, optix::float3 & radiance, int & casts_shadows, const HitInfo & data, unsigned int& seed)
{
    const optix::float3& hit_point = data.hit_point;
    const optix::float3& normal = data.hit_normal;

    // TODO: Now I am tracing two rays per sample == wasteful. Need to revactor a little bit
    // the prd_shadow pipeline to have a more efficient way of sampling the color of the enviroment
    // lighting. It will slow down other passes anyway (amybe have another ray type?)

    optix::uint& t = seed;

    optix::float3 color = optix::make_float3(0.0f);

    float zeta1 = rnd(t);
    float zeta2 = rnd(t);
    optix::float3 smp = sample_hemisphere_cosine(optix::make_float2(zeta1, zeta2), normal);
    float3 emission = optix::make_float3(0.0f);
    float V = trace_shadow_ray(hit_point, smp, scene_epsilon, RT_DEFAULT_MAX, emission);
    if (V == 1.0f) // I did not hit anything == environment light;
    {
        color += emission;
    }


    wi = normalize(-smp);
    radiance = color;
    casts_shadows = 1;
}

__device__ __inline__ void evaluate_area_light_no_sr(float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index, const float3& hit_point, const float3& normal, float tmin)
{
    //	assert(data != NULL);

    float zeta1 = rnd(seed);
    seed = lcg(seed);
    float zeta2 = rnd(seed);
    seed = lcg(seed);
    float zeta3 = rnd(seed);
    seed = lcg(seed);
    TriangleLight triangle = area_lights[static_cast<int>(area_lights.size()* zeta1)];
    optix::float3 point = sample_point_triangle(zeta2, zeta3, triangle.v1, triangle.v2, triangle.v3);
    optix::float3 to_light_un = point - hit_point;
    float dist_sq = dot(to_light_un, to_light_un);
    optix::float3 to_light = normalize(to_light_un);
    casts_shadows = 1;
    float3 color = triangle.emission * max(dot(to_light, normal), 0.0f) * max(dot(to_light, -triangle.normal), 0.0f) / dist_sq * triangle.area;
    radiance = area_lights.size() * color;
    wi = normalize(hit_point - point);
}

__device__ __inline__ void evaluate_direct_light(const float3& hit_point, const float3& normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index, float tmin = scene_epsilon)
{
    switch (light_type)
    {
    case LIGHT_TYPE_POINT: evaluate_point_light(hit_point, normal, wi, radiance, casts_shadows, seed, light_index); break;
    case LIGHT_TYPE_AREA: evaluate_area_light(hit_point, normal, wi, radiance, casts_shadows, seed, light_index);  break;
    default:
    case LIGHT_TYPE_SKY:
    case LIGHT_TYPE_DIR: evaluate_directional_light(hit_point, normal, wi, radiance, casts_shadows, seed, light_index); break;
    }
}



__device__ __forceinline__ int light_size()
{
    switch (light_type)
    {
    case LIGHT_TYPE_POINT: return point_light_size(); break;
    case LIGHT_TYPE_AREA: return area_light_size();  break;
    default:
    case LIGHT_TYPE_SKY:
    case LIGHT_TYPE_DIR: return directional_light_size(); break;
    }
}
