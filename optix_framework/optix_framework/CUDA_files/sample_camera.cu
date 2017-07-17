// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011



#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"

using namespace optix;
#define DIRLIGHT
#define USE_SHADOW_RAYS
// Triangle mesh data
rtBuffer<float3> sampling_vertex_buffer;
rtBuffer<float3> sampling_normal_buffer;
rtBuffer<int3>   sampling_vindex_buffer;
rtBuffer<int3>   sampling_nindex_buffer;
rtBuffer<float>  area_cdf;  
rtDeclareVariable(float, total_area, , );


// Window variables
rtBuffer<PositionSample> sampling_output_buffer;


rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim, rtLaunchDim, );
//rtDeclareVariable(uint, current_translucent_obj, , );
rtDeclareVariable(uint, frame, , );

__forceinline__ __device__ unsigned int cdf_bsearch(float xi)
{
  uint table_size = area_cdf.size();
  uint middle = table_size = table_size>>1;
  uint odd = 0;
  while(table_size > 0)
  {
    odd = table_size&1;
    table_size = table_size>>1;
    unsigned int tmp = table_size + odd;
    middle = xi > area_cdf[middle] ? middle + tmp : (xi < area_cdf[middle - 1] ? middle - tmp : middle);
  }
  return middle;
}

RT_PROGRAM void sample_camera()
{
  uint idx = launch_index;
  PositionSample& sample = sampling_output_buffer[idx];
  uint t = tea<16>(idx, frame);
  // sample a triangle
  uint triangles = sampling_vindex_buffer.size();
  uint sm = (int)(rnd(t) * triangles);
  //uint sm = cdf_bsearch(rnd(t));
  int3 idx_vxt = sampling_vindex_buffer[sm];
  float3 v0 = sampling_vertex_buffer[idx_vxt.x];
  float3 v1 = sampling_vertex_buffer[idx_vxt.y];
  float3 v2 = sampling_vertex_buffer[idx_vxt.z];
  float3 perp_triangle = cross(v1 - v0, v2 - v0);
  float area = 0.5*length(perp_triangle);

  // sample a point in the triangle
  float xi1 = sqrt(rnd(t));
  float xi2 = rnd(t);
  float u = 1.0f - xi1;
  float v = (1.0f - xi2)*xi1;
  float w = xi1*xi2;
  sample.pos = u*v0 + v*v1 + w*v2;

 
  // compute the sample normal
  if(sampling_normal_buffer.size() > 0)
  {
    int3 nidx_vxt = sampling_nindex_buffer[sm];
    float3 n0 = sampling_normal_buffer[nidx_vxt.x];
    float3 n1 = sampling_normal_buffer[nidx_vxt.y];
    float3 n2 = sampling_normal_buffer[nidx_vxt.z];
    sample.normal = normalize(u*n0 + v*n1 + w*n2);
  }
  else
    sample.normal = normalize(perp_triangle);

  // compute the cosine of the angle of incidence
 

  float3 light_vector;
  float3 light_radiance;
  int cast_shadows;
  uint s = 0;

#ifdef USE_SHADOW_RAYS
#ifdef POINTLIGHT
  uint light_idx = point_lights.size()*rnd(t);
  PointLight& light = point_lights[light_idx];
  float3 wi = light.position - sample.pos;
  float r_sqr = dot(wi, wi);
  float r = sqrt(r_sqr);
  wi /= r;
  float3 Le = light.intensity/r_sqr;

  PerRayData_shadow shadow_prd;
  shadow_prd.attenuation = 1.0f;
  Ray shadow_ray(sample.pos, wi, shadow_ray_type, scene_epsilon, r);
  rtTrace(top_shadower, shadow_ray, shadow_prd);
  float V = shadow_prd.attenuation;

#elif defined(DIRLIGHT)
  uint light_idx = directional_lights.size()*rnd(t);
  DirectionalLight& light = directional_lights[light_idx];
  float3 wi = -light.direction;
  float r = RT_DEFAULT_MAX;
  float3 Le = light.emission;

  PerRayData_shadow shadow_prd;
  shadow_prd.attenuation = 1.0f;
  Ray shadow_ray(sample.pos, wi, shadow_ray_type, scene_epsilon, r);
  rtTrace(top_shadower, shadow_ray, shadow_prd);
  float V = shadow_prd.attenuation;
#else
  float3 wi, Le; int sh; unsigned int seed = 0;
  float r = RT_DEFAULT_MAX;
  uint light_idx = 0;
  evaluate_area_light_no_sr(wi, Le, sh, seed, light_idx, sample.pos, sample.normal, 0);
  float V = 1.0f;
  wi = -wi;
#endif  
  // trace a shadow ray to compute the visibility term
  
#else


  PerRayData_radiance prd;
  prd.importance = 1.0f;
  prd.depth = 0;
  prd.seed = launch_index;
  prd.flags = 0;
  prd.flags &= ~(RayFlags::HIT_DIFFUSE_SURFACE); //Just for clarity
  prd.flags |= RayFlags::USE_EMISSION;
  prd.result = make_float3(0.0f);

  float x1 = rnd(t);
  float x2 = rnd(t);
  float3 wi = sample_hemisphere_cosine(make_float2(x1, x2), sample.normal);
  Ray ray(sample.pos, wi, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
  rtTrace(top_object, ray, prd);
  float V = 1.0f;
  float3 Le = prd.result;

#endif

  sample.dir = wi;

  // Compute transmitted radiance
  //sample.L = T12*V*Le*cos_theta_i*total_area;
  sample.L = Le*make_float3(triangles*area);
  //printf("Le: %f, L: %f\n", Le.x, sample.L.x);
}
