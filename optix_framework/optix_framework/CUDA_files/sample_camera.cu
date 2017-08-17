// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011



#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"
#include "device_environment_map.h"

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
rtDeclareVariable(unsigned int, N, , );

// Window variables
rtBuffer<PositionSample> sampling_output_buffer;

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
    uint idx = launch_index.x;
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

    float3 Le = make_float3(0.0f);
    uint lights = light_size();
    uint light_idx = (lights + 1)*rnd(t);

    if (light_idx == lights)
    {
        for (int j = 0; j < N; j++)
        {
            sample_environment(light_vector, light_radiance, HitInfo(sample.pos, sample.normal), t);
            float cos_theta = dot(light_vector, sample.normal);
            if (cos_theta <= 0.0) continue;
            Le += light_radiance;
        }
        Le /= static_cast<float>(N);
    }
    else
    {
        for (int j = 0; j < N; j++)
        {
            evaluate_direct_light(sample.pos, sample.normal, light_vector, light_radiance, cast_shadows, t, light_idx);
            Le += light_radiance;
        }

        Le /= static_cast<float>(N);
    }
        
    sample.dir = light_vector;

    // Compute transmitted radiance
    //sample.L = T12*V*Le*cos_theta_i*total_area;
    sample.L = Le*make_float3(triangles*area);
    //printf("Le: %f, L: %f\n", Le.x, sample.L.x);
}
