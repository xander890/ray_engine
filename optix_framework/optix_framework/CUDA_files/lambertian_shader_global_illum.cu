#include <optix_device.h>
#include <optix_math.h>
#include "../random.h"
#include "..\point_light.h"
#include "..\structs.h"

using namespace optix;

#define GATHER

// Standard ray variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// Variables for shading
rtBuffer<PointLight> lights;
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

// Material properties (corresponding to OBJ mtl params)
rtTextureSampler<float4, 2> ambient_map;
rtTextureSampler<float4, 2> diffuse_map; 

// Shadow variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );

#ifdef GATHER
// Recursive ray tracing variables
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
#endif

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = 0.0f;
  rtTerminateRay();
}

// Given a direction vector v sampled on the hemisphere
// over a surface point with the z-axis as its normal,
// this function applies the same rotation to v as is
// needed to rotate the z-axis to the actual normal
// [Frisvad, Journal of Graphics Tools 16, 2012].
__inline__ __device__ void rotate_to_normal(const float3& normal, float3& v)
{
	if(normal.z < -0.999999f)
  {
    v = make_float3(-v.y, -v.x, -v.z);
    return;
  }
  float a = 1.0f/(1.0f + normal.z);
  float b = -normal.x*normal.y*a;
  v =   make_float3(1.0f - normal.x*normal.x*a, b, -normal.x)*v.x 
      + make_float3(b, 1.0f - normal.y*normal.y*a, -normal.y)*v.y 
      + normal*v.z;
}

__inline__ __device__ float3 sample_cosine_weighted(float3 normal, uint t)
{
  // Get random numbers
  float cos_theta = sqrt(rnd(t));
	float phi = 2.0*M_PIf*rnd(t);

	// Calculate new direction as if the z-axis were the normal
  float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
  float3 v = make_float3(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;
}

// Closest hit program for Lambertian shading using the basic light as a directional source.
// This one includes shadows.
RT_PROGRAM void lambertian_shader() 
{ 
  float3 hit_pos = ray.origin + t_hit * ray.direction; 
  float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); 
  float3 ffnormal = faceforward(normal, -ray.direction, normal); 
  float3 emission = make_float3(tex2D(ambient_map, texcoord.x, texcoord.y));
  float3 rho_d = make_float3(tex2D(diffuse_map, texcoord.x, texcoord.y));

  // Emission
  float3 result = emission; 

  // Direct illumination
  for(int i = 0; i < lights.size(); ++i) 
  { 
    PointLight& light = lights[i];
    float3 wi = light.position - hit_pos;
    float r_sqr = dot(wi, wi);
    float r = sqrt(r_sqr);
    wi /= r;
    float cos_theta = dot(ffnormal, wi);
    if(cos_theta > 0.0)
    {
      float V = 1.0f;
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = 1.0f;
      Ray shadow_ray(hit_pos, wi, shadow_ray_type, scene_epsilon, r);
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      V = shadow_prd.attenuation;
      float3 Li = V*light.intensity/r_sqr;
      result += Li*rho_d*M_1_PIf*cos_theta; 
    }
  }

#ifdef GATHER
  // Indirect illumination (final gathering)
  if(prd_radiance.depth < 1)
  {
    float3 indirect = make_float3(0.0f);
    for(unsigned int i = 0; i < 10; ++i)
    {
      PerRayData_radiance prd_new;
      prd_new.depth = prd_radiance.depth + 1;
      float3 new_dir = sample_cosine_weighted(ffnormal, prd_radiance.seed);
      Ray new_ray(hit_pos, new_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, new_ray, prd_new);
      indirect += prd_new.result;
    }
    result += indirect*rho_d/10.0f;
  }
#endif
  prd_radiance.result = result; 
}
