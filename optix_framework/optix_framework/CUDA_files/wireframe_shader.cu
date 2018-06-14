// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>
#include <device_mesh_data.h>
#include <camera.h>
using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(CameraData, camera_data, , );
// Variables for shading
rtDeclareVariable(float2, barys, attribute barys, ); 
rtDeclareVariable(int, primitive, attribute primitive, );
// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { rtTerminateRay(); }

_fn float point_line_distance(const optix::float3 & point, const optix::float3 & p1, const optix::float3 & p2)
{
	const float3 n = normalize(p2 - p1);
	const float3 ap = p1 - point;
	return length(ap - dot(ap, n) * n);
}

_fn float3 to_ws(const optix::float3 & object_space)
{
	return rtTransformPoint(RT_OBJECT_TO_WORLD, object_space);
}

__device__ __forceinline__ float2 to_norm_screen_coordinates(const optix::float3 & p)
{
	float3 d = normalize(p - camera_data.eye);
	float3 ip_c = camera_data.inv_view_matrix * d;
	ip_c /= ip_c.z;
	return optix::make_float2(ip_c) * 0.5f + make_float2(0.5f);
}

// Closest hit program for drawing shading normals
RT_PROGRAM void shade()
{

	int3 v_idx = vindex_buffer[primitive];

	float3 p0 = to_ws(vertex_buffer[v_idx.x]);
	float3 p1 = to_ws(vertex_buffer[v_idx.y]);
	float3 p2 = to_ws(vertex_buffer[v_idx.z]);

	float3 hit_pos = ray.origin + t_hit * ray.direction;
	float3 distances;
	distances.x = point_line_distance(hit_pos, p0, p1);
	distances.y = point_line_distance(hit_pos, p1, p2);
	distances.z = point_line_distance(hit_pos, p2, p0);
	float d = fminf(distances);
	const float border = 0.25f / fmaxf(make_float2(launch_dim));
	if (d < border)
		prd_radiance.result = make_float3(1, 0, 0);
	else
		prd_radiance.result = make_float3(1);
}