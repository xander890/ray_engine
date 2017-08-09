// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include "../math_helpers.h"
#include "../color_helpers.h"
#include "merl_common.h"

using namespace optix;

rtDeclareVariable(int, environment_map_tex_id, , ) = 0;
rtDeclareVariable(Matrix3x3, lightmap_rotation_matrix, , );

// Output buffers
rtBuffer<float, 2> env_luminance;
rtBuffer<float> marginal_f;
rtBuffer<float> marginal_pdf;
rtBuffer<float, 2> conditional_pdf;
rtBuffer<float> marginal_cdf;
rtBuffer<float, 2> conditional_cdf;

RT_PROGRAM void env_luminance_camera()
{
  float2 uv = (make_float2(launch_index) + 0.5f)/make_float2(launch_dim);
  float theta = uv.y*M_PIf;
  float phi = uv.x*2.0f*M_PIf;
  float sin_theta, cos_theta, sin_phi, cos_phi;
  sincosf(theta, &sin_theta, &cos_theta);
  sincosf(phi, &sin_phi, &cos_phi);
  float3 dir = make_float3(sin_theta*sin_phi, -cos_theta, -sin_theta*cos_phi);
  float2 uv2 = direction_to_uv_coord_cubemap(dir, lightmap_rotation_matrix);
  float3 texel = make_float3(rtTex2D<float4>(environment_map_tex_id, uv2.x, uv2.y));
  env_luminance[launch_index] = luminance_NTSC(texel)*sin_theta;
}

RT_PROGRAM void env_marginal_camera()
{
  if(launch_index.x == 0)
  {
    float c_f_sum = 0.0f;
    for(uint i = 0; i < launch_dim.x; ++i)
    {
      uint2 idx = make_uint2(i, launch_index.y);
      c_f_sum += env_luminance[idx];
    }
    marginal_f[launch_index.y] = c_f_sum/launch_dim.x;
  }
}

RT_PROGRAM void env_pdf_camera()
{
  conditional_pdf[launch_index] = env_luminance[launch_index]/marginal_f[launch_index.y];
  float cdf_sum = 0.0f;
  for(uint i = 0; i <= launch_index.x; ++i)
  {
    uint2 idx = make_uint2(i, launch_index.y);
    cdf_sum += env_luminance[idx];
  }
  cdf_sum /= launch_dim.x;
  conditional_cdf[launch_index] = cdf_sum/marginal_f[launch_index.y];
  if(launch_index == launch_dim - 1)
    conditional_cdf[launch_index] = 1.0f;  // handle numerical instability

  if(launch_index.x == 0)
  {
    float m_f_sum = 0.0f;
    for(uint i = 0; i < marginal_f.size(); ++i)
    {
      m_f_sum += marginal_f[i];
      if(i == launch_index.y)
        cdf_sum = m_f_sum;
    }
    m_f_sum /= launch_dim.y;
    cdf_sum /= launch_dim.y;
    marginal_pdf[launch_index.y] = marginal_f[launch_index.y]/m_f_sum;
    marginal_cdf[launch_index.y] = cdf_sum/m_f_sum;
    if(launch_index.y == launch_dim.y - 1)
      marginal_cdf[launch_index.y] = 1.0f; // handle numerical instability
  }
}
