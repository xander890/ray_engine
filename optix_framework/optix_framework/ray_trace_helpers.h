#ifndef  RAY_TRACE_HELPERS
#define RAY_TRACE_HELPERS
#include "random.h"
#include "sampling_helpers.h"
#include "optical_helper.h"
#include "structs.h"

__inline__ __device__ PerRayData_radiance get_empty()
{
	PerRayData_radiance r;
	r.colorband = -1;
	r.result = make_float3(0.0f);
	r.flags = 0;
	r.seed = 0;
	r.depth = 0;
	return r;
}

__inline__ __device__ PerRayData_radiance init_copy(PerRayData_radiance & to_copy)
{
	PerRayData_radiance r;
	r.colorband = to_copy.colorband;
	r.result = to_copy.result;
	r.flags = to_copy.flags;
	r.seed = to_copy.seed;
	r.depth = to_copy.depth;
	return r;
}

__inline__ __device__ PerRayData_radiance prepare_new_pt_payload(PerRayData_radiance & to_copy)
{
	PerRayData_radiance r;
	r.colorband = to_copy.colorband;
	r.result = make_float3(0);
	r.flags = to_copy.flags | RayFlags::USE_EMISSION;
	r.seed = to_copy.seed;
	r.depth = to_copy.depth + 1;
	return r;
}


__inline__ __device__ float trace_shadow_ray(const optix::float3 & hit_pos, const optix::float3 & direction, float tmin, float tmax, optix::float3 & emission)
{
	PerRayData_shadow shadow_prd;
	shadow_prd.attenuation = 1.0f;
	shadow_prd.emission = optix::make_float3(0.0f);
	optix::Ray shadow_ray = optix::make_Ray(hit_pos, direction, RAY_TYPE_SHADOW, tmin, tmax);
	optix::Ray ray = shadow_ray;

	rtTrace(top_object, shadow_ray, shadow_prd);
	emission = shadow_prd.emission;
	return shadow_prd.attenuation;
}

namespace optix
{
	__forceinline__ __device__ optix::float3 fpowf(const optix::float3 & p, const float ex)
	{
		return optix::make_float3(powf(p.x, ex), powf(p.y, ex), powf(p.z, ex));
	}

}
__inline__ __device__ float trace_shadow_ray(const optix::float3 & hit_pos, const optix::float3 & direction, float tmin, float tmax)
{
	float3 emission;
	return trace_shadow_ray(hit_pos, direction, tmin, tmax, emission);
}



__device__ __inline__ void get_glass_rays(const optix::float3& wo, const float ior, const float3& hit_pos, float3& normal, optix::Ray& reflected_ray, optix::Ray& refracted_ray, float& R, float& cos_theta_signed)
{
	// Compute Fresnel reflectance
	cos_theta_signed = dot(normal, -wo);
	float eta = cos_theta_signed < 0.0f ? 1.0f / ior : ior;
	float recip_eta = 1.0f / eta;
	normal = normal*copysignf(1.0f, cos_theta_signed);
	float cos_theta = fabsf(cos_theta_signed);
	float sin_theta_t_sqr = recip_eta*recip_eta*(1.0f - cos_theta*cos_theta);
	float cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
	R = sin_theta_t_sqr < 1.0f ? fresnel_R(cos_theta, cos_theta_t, eta) : 1.0f;

	float3 refr_dir = recip_eta*wo + normal*(recip_eta*cos_theta - cos_theta_t);
	refracted_ray = optix::make_Ray(hit_pos, refr_dir, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);

	float3 reflected_dir = reflect(wo, normal);
	reflected_ray = optix::make_Ray(hit_pos, reflected_dir, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
}


__device__ __inline__ void get_glass_rays(const optix::Ray& ray, const float ior, const float3& hit_pos, float3& normal, optix::Ray& reflected_ray, optix::Ray& refracted_ray, float& R, float& cos_theta_signed)
{
	get_glass_rays(ray.direction, ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta_signed);
}



__device__ __inline__ void sample_light(const float3& position, const float3 & normal, const uint& ray_depth, uint& seed, float3 & wi, float3 & Li)
{
	float zeta1 = rnd(seed);
	float zeta2 = rnd(seed);
	optix::float3 smp = sample_hemisphere_cosine(optix::make_float2(zeta1, zeta2), normal);
	wi = normalize(smp);

	PerRayData_radiance prd = get_empty();
	prd.flags = RayFlags::USE_EMISSION;
	prd.depth = ray_depth+1;
	prd.seed = seed;
	optix::Ray ray = optix::make_Ray(position, wi, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);

	rtTrace(top_object, ray, prd);
	seed = prd.seed;
	Li = prd.result * M_PIf;
}


#endif