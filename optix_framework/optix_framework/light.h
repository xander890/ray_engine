#pragma once
#include "device_common_data.h"
#include <singular_light.h>
#include <area_light.h>
#include <ray_trace_helpers.h>
#include "host_device_common.h"

// Helper functions and data structures for lights. Note that every time a light is used this header must be included.

// Directional lights
rtBuffer<SingularLightData, 1> singular_lights;
// Area light code
rtBuffer<TriangleLight, 1> area_lights;

rtDeclareVariable(int, light_type, , );
rtDeclareVariable(unsigned int, importance_sample_area_lights, , ) = 0;

__forceinline__ __device__
int no_light_size() { return 1; }

__forceinline__ __device__
void evaluate_no_light(const float3& test, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index) { radiance = make_float3(1, 0, 0); }

__forceinline__ __device__
int singular_light_size() { return singular_lights.size(); }

__forceinline__ __device__
void evaluate_singular_light(const float3 & hit_point, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, unsigned int& seed, unsigned int& light_index)
{
    SingularLightData l = singular_lights[light_index];
    wi = (l.type == LightType::POINT)? l.direction - hit_point  : -l.direction ;
    wi = normalize(wi);

    float V = 1.0f;

    if (l.casts_shadow)
    {
        V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX);
    }

    float atten = (l.type == LightType::POINT) ? dot(wi, wi) : 1.0f;
    radiance = V * l.emission / atten;
    casts_shadows = l.casts_shadow;
}

__forceinline__ __device__
int area_light_size() { return area_lights.size() > 0? 1 : 0; } // This means that we will randomly sample from triangle lights instead of going though all of them.

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
    case LightType::AREA:  evaluate_area_light(hit_point, normal, wi, radiance, casts_shadows, seed, light_index);  break;
    default:
    case LightType::SKY:
    case LightType::POINT:
    case LightType::DIRECTIONAL:   evaluate_singular_light(hit_point, normal, wi, radiance, casts_shadows, seed, light_index); break;
    }
}



__device__ __forceinline__ int light_size()
{
    switch (light_type)
    {
    case LightType::AREA: return area_light_size();  break;
    default:
    case LightType::SKY:
	case LightType::POINT: 
    case LightType::DIRECTIONAL: return singular_light_size(); break;
    }
}

__device__ __inline__ void sample_light(const float3& position, const float3 & normal, const uint& ray_depth, uint& seed, float3 & wi, float3 & Li)
{
	if (importance_sample_area_lights == 1)
	{
		float zeta1 = rnd(seed);
		float zeta2 = rnd(seed);
		optix::float3 smp = sample_hemisphere_cosine(optix::make_float2(zeta1, zeta2), normal);
		wi = normalize(smp);

		PerRayData_radiance prd = get_empty();
		prd.flags = RayFlags::USE_EMISSION;
		prd.depth = ray_depth + 1;
		prd.seed = seed;
		optix::Ray ray = optix::make_Ray(position, wi,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);

		rtTrace(top_object, ray, prd);
		seed = prd.seed;
		Li = prd.result * M_PIf;
	}
	else
	{
		int casts_shadows = 0;
		unsigned int light_index;
		evaluate_direct_light(position, normal, wi, Li, casts_shadows, seed, light_index);
	}
}