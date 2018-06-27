#pragma once
#include "random_device.h"
#include "sampling_helpers.h"
#include "optics_utils.h"
#include "structs.h"

_fn PerRayData_radiance get_empty()
{
	PerRayData_radiance r;
	r.colorband = -1;
	r.result = optix::make_float3(0.0f);
	r.flags = 0;
	r.sampler = nullptr;
	r.depth = 0;
	return r;
}

_fn PerRayData_radiance init_copy(PerRayData_radiance & to_copy)
{
	PerRayData_radiance r;
	r.colorband = to_copy.colorband;
	r.result = to_copy.result;
	r.flags = to_copy.flags;
	r.sampler = to_copy.sampler;
	r.depth = to_copy.depth;
	return r;
}

_fn PerRayData_radiance prepare_new_pt_payload(PerRayData_radiance & to_copy)
{
	PerRayData_radiance r;
	r.colorband = to_copy.colorband;
	r.result = optix::make_float3(0);
	r.flags = to_copy.flags | RayFlags::USE_EMISSION;
	r.sampler = to_copy.sampler;
	r.depth = to_copy.depth + 1;
	return r;
}


_fn float trace_shadow_ray(const optix::float3 & hit_pos, const optix::float3 & direction, float tmin, float tmax, optix::float3 & emission)
{
	PerRayData_shadow shadow_prd;
	shadow_prd.attenuation = 1.0f;
	shadow_prd.emission = optix::make_float3(0.0f);
	optix::Ray shadow_ray = optix::make_Ray(hit_pos, direction,  RayType::SHADOW, tmin, tmax);
	optix::Ray ray = shadow_ray;

	rtTrace(top_object, shadow_ray, shadow_prd);
	emission = shadow_prd.emission;
	return shadow_prd.attenuation;
}

namespace optix
{
	_fn optix::float3 fpowf(const optix::float3 & p, const float ex)
	{
		return optix::make_float3(powf(p.x, ex), powf(p.y, ex), powf(p.z, ex));
	}

}
_fn float trace_shadow_ray(const optix::float3 & hit_pos, const optix::float3 & direction, float tmin, float tmax)
{
	optix::float3 emission;
	return trace_shadow_ray(hit_pos, direction, tmin, tmax, emission);
}



_fn void get_glass_rays(const optix::float3& wo, const float ior, const optix::float3& hit_pos, const optix::float3& old_normal, optix::float3& normal, optix::Ray& reflected_ray, optix::Ray& refracted_ray, float& R, float& cos_theta_signed)
{
	normal = old_normal;
	cos_theta_signed = optix::dot(normal, wo);
    float eta;

    if(cos_theta_signed > 0.0f)
    {
        eta = 1.0f / ior;
    }
    else
    {
        normal = -normal;
        eta = ior;
    }

	optix::float3 refr_dir;
	refract(wo, normal, eta, refr_dir, R);

	refracted_ray = optix::make_Ray(hit_pos, refr_dir,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
	optix::float3 reflected_dir = optix::reflect(-wo, normal);
	reflected_ray = optix::make_Ray(hit_pos, reflected_dir,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
}

_fn void shadow_hit(PerRayData_shadow & shadow_payload, optix::float3 & emission)
{
    if (!(emission.x + emission.y + emission.z > 0.0f))
    {
        shadow_payload.attenuation = 0.0f;
    }

    rtTerminateRay();
}