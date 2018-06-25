// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011


#include <device_common.h>
#include <random_device.h>
#include <environment_map.h>
#include <camera_common.h>

using namespace optix;

// Camera variables
rtDeclareVariable(CameraData,   camera_data, , );
rtDeclareVariable(optix::uint4, render_bounds, , ); // Bounds, in case we want to limit what to render on the screen.

// Window variables
rtBuffer<float4, 2> output_buffer;

_fn bool check_bounds()
{
	return	launch_index.x >= render_bounds.x && launch_index.x < render_bounds.x + render_bounds.z &&
		launch_index.y >= render_bounds.y && launch_index.y < render_bounds.y + render_bounds.w;
}

_fn void trace(const Ray& ray, PerRayData_radiance & prd)
{
	rtTrace(top_object, ray, prd);

	if (isfinite(prd.result.x) && isfinite(prd.result.y) && isfinite(prd.result.z))
	{
		float4 curr_sum = (frame != 0) ? output_buffer[launch_index] * ((float)frame) : make_float4(0.0f);
		output_buffer[launch_index] = (make_float4(prd.result, 1.0f) + curr_sum) / ((float)(frame + 1));
		output_buffer[launch_index].w = 1.0f;
		optix_print("Final color: %f %f %f, sum = %f %f %f\n", prd.result.x, prd.result.y, prd.result.z, output_buffer[launch_index].x, output_buffer[launch_index].y, output_buffer[launch_index].z);
	}
}

_fn PerRayData_radiance get_starting_payload(TEASampler * sampler)
{
	PerRayData_radiance prd;
	prd.depth = 0;
	prd.sampler = sampler;
	prd.flags = 0;
	prd.flags |= RayFlags::USE_EMISSION;
	prd.colorband = -1;
	prd.result = make_float3(0);
	return prd;
}

RT_PROGRAM void pinhole_camera()
{
    optix_print("Frame %d!\n", frame);
	if (check_bounds())
	{
		TEASampler sampler(launch_dim.x*launch_index.y + launch_index.x, frame);
		PerRayData_radiance prd = get_starting_payload(&sampler);
		float2 jitter = sampler.next2D() * camera_data.downsampling;
		uint2 real_pixel = launch_index * camera_data.downsampling;
		float2 ip_coords = (make_float2(real_pixel) + jitter) / make_float2(launch_dim) * 2.0f - 1.0f;

	#ifdef ORTHO
		float3 direction = normalize(W);
		float3 origin = mEye + ip_coords.x*U + ip_coords.y*V;
	#else
		float3 origin = camera_data.eye;
		float3 direction = normalize(ip_coords.x*camera_data.U + ip_coords.y*camera_data.V + camera_data.W);
	#endif
		Ray ray(origin, direction,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
		trace(ray, prd);
	}
	else
	{
		output_buffer[launch_index] = make_float4(0);
	}

}


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_float4(0.0, 0.0, 100000.0, 1.0f);
  rtPrintExceptionDetails();

}

RT_PROGRAM void empty() {}