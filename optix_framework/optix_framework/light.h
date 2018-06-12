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

rtDeclareVariable(unsigned int, importance_sample_area_lights, , ) = 0;

_fn
int no_light_size() { return 1; }

_fn
void evaluate_no_light(const float3& test, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, TEASampler * sampler, unsigned int& light_index) { radiance = make_float3(1, 0, 0); }

_fn
int singular_light_size() { return singular_lights.size(); }

_fn
void evaluate_singular_light(const float3 & hit_point, const float3 & hit_normal, float3& wi, float3 & radiance, int & casts_shadows, TEASampler * sampler, unsigned int& light_index)
{
    if(singular_light_size() == 0)
    {
        light_index = 0;
        radiance = make_float3(0);
        wi = hit_normal;
        casts_shadows = 0;
        return;
    }

    light_index = int(sampler->next1D() * singular_light_size());
    SingularLightData l = singular_lights[light_index];
    wi = (l.type == LightType::POINT)? l.direction - hit_point  : -l.direction ;
    wi = normalize(wi);

    float V = 1.0f;

    if (l.casts_shadow == 1)
    {
        V = trace_shadow_ray(hit_point, wi, scene_epsilon, RT_DEFAULT_MAX);
    }

    float atten = (l.type == LightType::POINT) ? dot(wi, wi) : 1.0f;
    radiance = V * l.emission / atten;
    radiance *= singular_light_size(); // Pdf of choosing one of the lights.
    casts_shadows = l.casts_shadow;
}

_fn
int area_light_size() { return area_lights.size() > 0? 1 : 0; } // This means that we will randomly sample from triangle lights instead of going though all of them.

_fn void evaluate_area_light(const float3& hit_point, const float3& normal, float3& wi, float3 & radiance, int & casts_shadows, TEASampler * sampler, unsigned int& light_index, float tmin = scene_epsilon)
{
    //	assert(data != NULL);
    float zeta1 = sampler->next1D();
    float zeta2 = sampler->next1D();
    float zeta3 = sampler->next1D();
    int idx = clamp((int)(area_lights.size() * zeta1), 0, area_lights.size() - 1);
    TriangleLight triangle = area_lights[idx];
    optix::float3 point = sample_point_triangle(zeta2, zeta3, triangle.v1, triangle.v2, triangle.v3);
    optix::float3 to_light_un = point - hit_point;
    float dist_sq = dot(to_light_un, to_light_un);
    float dist = sqrt(dist_sq);
    wi = to_light_un / dist;
    float V = trace_shadow_ray(hit_point, wi, tmin, dist - scene_epsilon);
    casts_shadows = 1;
    radiance = V * triangle.emission * max(dot(wi, normal), 0.0f) * max(dot(-wi, triangle.normal), 0.0f) / dist_sq * triangle.area * area_lights.size();
}

_fn void evaluate_direct_light(const float3& hit_point, const float3& normal, float3& wi, float3 & radiance, int & casts_shadows, TEASampler * sampler, unsigned int& light_index, float tmin = scene_epsilon)
{
    float area_probability = 1.0f;
    float singular_probability = 1.0f;

    if(area_light_size() == 0)
        area_probability = 0.0f;
    if(singular_light_size() == 0)
        singular_probability = 0.0f;

    float joint = area_probability + singular_probability;

    if(joint == 0.0f)
    {
        radiance = make_float3(0.0f);
        wi = normal;
        casts_shadows = 1;
        light_index = 0;
        return;
    }

    area_probability /= joint;

    if(sampler->next1D() < area_probability)
    {
        evaluate_area_light(hit_point, normal, wi, radiance, casts_shadows, sampler, light_index);
        radiance /= area_probability;
    }
    else
    {
        evaluate_singular_light(hit_point, normal, wi, radiance, casts_shadows, sampler, light_index);
        radiance /= 1 - area_probability;
    }
}


_fn void sample_light(const float3& position, const float3 & normal, const uint& ray_depth, TEASampler* sampler, float3 & wi, float3 & Li)
{
	if (importance_sample_area_lights == 0)
	{
		optix::float3 smp = sample_hemisphere_cosine(sampler->next2D(), normal);
		wi = normalize(smp);

		PerRayData_radiance prd = get_empty();
		prd.flags = RayFlags::USE_EMISSION;
		prd.depth = ray_depth + 1;
		prd.sampler = sampler;
		optix::Ray ray = optix::make_Ray(position, wi,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);

		rtTrace(top_object, ray, prd);
		Li = prd.result * M_PIf;
	}
	else
	{
		int casts_shadows = 0;
		unsigned int light_index;
		evaluate_direct_light(position, normal, wi, Li, casts_shadows, sampler, light_index);
	}
}