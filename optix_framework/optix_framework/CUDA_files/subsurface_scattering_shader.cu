// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011


#include <device_common_data.h>
#include "../math_helpers.h"
#include "../random.h"
#include "../directional_dipole.h"
#include "../optical_helper.h"
#include "../structs.h"
#include <ray_trace_helpers.h>

using namespace optix;

//#define REFLECT

// Standard ray variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// SS properties
rtDeclareVariable(ScatteringMaterialProperties, scattering_properties, , );

// Variables for shading
rtBuffer<PositionSample> sampling_output_buffer;
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

rtDeclareVariable(unsigned int, bssrdf_enabled, , );

#ifdef REFLECT
// Recursive ray tracing variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
#endif

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = 0.0f;
  rtTerminateRay();
}

// Closest hit program for Lambertian shading using the basic light as a directional source
RT_PROGRAM void shade() 
{

	if (bssrdf_enabled == 0u)
	{
		prd_radiance.result = make_float3(0);
		return;
	}

  float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); 
  float3 xo = ray.origin + t_hit * ray.direction; 
  float3 wo = -ray.direction;
  float3 no = faceforward(n, wo, n);
  float cos_theta_o = dot(wo, no);
  
  float3 accumulate = make_float3(0.0f);
  uint scattering_samples = sampling_output_buffer.size();
  uint t = prd_radiance.seed;
  
  const ScatteringMaterialProperties& props = scattering_properties;
  const float ior = props.indexOfRefraction;
  const float recip_ior = 1.0f / ior;
  //optix_print("Interface ior: %f\n", ior);
  //printvec3(props.absorption);
  //printvec3(props.scattering);

  const float chosen_transport_rr = min(props.transport.x, min(props.transport.y, props.transport.z));
  Ray reflected_ray, refracted_ray;
  float R, cos_theta;
  get_glass_rays(ray, ior, xo, no, reflected_ray, refracted_ray, R, cos_theta);

  const float T21 = 1.0f - R;

  for (uint i = 0; i < scattering_samples; ++i)
  {
     const PositionSample& sample = sampling_output_buffer[i];
   
    // compute contribution if sample is non-zero
    if(dot(sample.L, sample.L) > 0.0f)
    {
      // Russian roulette
      float exp_term = exp(-length(xo - sample.pos) * chosen_transport_rr); 
      if(rnd(t) < exp_term)
      {
		  const float3 wi = sample.dir;
		  const float cos_theta_i = max(dot(wi, sample.normal), 0.0f);
		  const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
		  const float sin_theta_t_sqr = recip_ior*recip_ior*(1.0f - cos_theta_i_sqr);
		  const float cos_theta_t = sqrt(1.0f - sin_theta_t_sqr);
		  const float3 w12 = recip_ior*(cos_theta_i*sample.normal - wi) - sample.normal*cos_theta_t;
		  const float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_t, ior);

		  accumulate += cos_theta_i * T12 * sample.L * bssrdf(sample.pos, sample.normal, w12, xo, no, props) / exp_term;
      }
    }
  }

    // Direct transmission, Delta eddington approximation
	PerRayData_radiance prd_new_ray;
	prd_new_ray.depth = prd_radiance.depth + 1;
	prd_new_ray.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
	float3 color_direct_transmission = make_float3(0);
	float xi = rnd(t);
	prd_new_ray.seed = t;
	if (xi < R)
	{
		rtTrace(top_object, reflected_ray, prd_new_ray);
	}
	else
	{
		rtTrace(top_object, refracted_ray, prd_new_ray);
	}

	color_direct_transmission = prd_new_ray.result;

	// Modified sigma_t tilde from the delta-eddington approximation.
	float gs = props.meancosine.x;// / (props.meancosine.x + 1);
	const float3 delta_edd_coefficient = props.scattering * (1 -gs * gs) + props.absorption;

	color_direct_transmission *= cos_theta < 0.0f ? expf(-delta_edd_coefficient*t_hit) : make_float3(1.0f);

	prd_radiance.result = T21*(accumulate*props.global_coeff/(float)scattering_samples); 
	prd_radiance.seed = t;

#ifdef REFLECT
  // Trace reflected ray
  if(prd_radiance.depth < 2)
  {
    float3 reflected_dir = 2.0f*cos_theta_o*no - wo;
    PerRayData_radiance prd_reflected;
    prd_reflected.depth = prd_radiance.depth + 1;
    Ray reflected(xo, reflected_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(top_object, reflected, prd_reflected);
    prd_radiance.result += R*prd_reflected.result;
  }
#endif
}
