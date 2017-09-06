// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011


#include <device_common_data.h>
#include <random.h>
#include <environment_map.h>
#include <camera.h>

using namespace optix;

// Camera variables
rtDeclareVariable(CameraData,   camera_data, , );

// Window variables
rtBuffer<float4, 2> output_buffer;

//#define TIME_VIEW

__forceinline__ __device__ bool check_bounds()
{
	return	launch_index.x >= camera_data.render_bounds.x && launch_index.x < camera_data.render_bounds.x + camera_data.render_bounds.z &&
		launch_index.y >= camera_data.render_bounds.y && launch_index.y < camera_data.render_bounds.y + camera_data.render_bounds.w;
}

__forceinline__ __device__ void trace(const Ray& ray, PerRayData_radiance & prd)
{
	rtTrace(top_object, ray, prd);

	if (isfinite(prd.result.x) && isfinite(prd.result.y) && isfinite(prd.result.z))
	{
		float4 curr_sum = (frame != 0) ? output_buffer[launch_index] * ((float)frame) : make_float4(0.0f);
		output_buffer[launch_index] = (make_float4(prd.result, 0.0f) + curr_sum) / ((float)(frame + 1));
		optix_print("Final color: %f %f %f, sum = %f %f %f\n", prd.result.x, prd.result.y, prd.result.z, output_buffer[launch_index].x, output_buffer[launch_index].y, output_buffer[launch_index].z);

		if (debug_index.x == launch_index.x && debug_index.y == launch_index.y)
		{
			output_buffer[launch_index] = make_float4(1, 0, 0, 1);
		}
	}
}

__forceinline__ __device__ PerRayData_radiance get_starting_payload()
{
	PerRayData_radiance prd;
	prd.depth = 0;
	prd.seed = tea<16>(launch_dim.x*launch_index.y + launch_index.x, frame);
	prd.flags = 0;
	prd.flags &= ~(RayFlags::HIT_DIFFUSE_SURFACE); //Just for clarity
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
		PerRayData_radiance prd = get_starting_payload();
		float2 jitter = make_float2(rnd(prd.seed), rnd(prd.seed)) * camera_data.downsampling;
		uint2 real_pixel = launch_index * camera_data.downsampling + make_uint2(camera_data.rendering_rectangle.x, camera_data.rendering_rectangle.y);
		float2 ip_coords = (make_float2(real_pixel) + jitter) / make_float2(camera_data.camera_size) * 2.0f - 1.0f;

	#ifdef ORTHO
		float3 direction = normalize(W);
		float3 origin = eye + ip_coords.x*U + ip_coords.y*V;
	#else
		float3 origin = camera_data.eye;
		float3 direction = normalize(ip_coords.x*camera_data.U + ip_coords.y*camera_data.V + camera_data.W);
	#endif
		Ray ray(origin, direction, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
		trace(ray, prd);
	}
	else
	{
		output_buffer[launch_index] = make_float4(0);
	}

}


RT_PROGRAM void pinhole_camera_w_matrix()
{
	if (check_bounds())
	{
		PerRayData_radiance prd = get_starting_payload();
		float2 jitter = make_float2(rnd(prd.seed), rnd(prd.seed))* camera_data.downsampling;
		uint2 real_pixel = launch_index * camera_data.downsampling + make_uint2(camera_data.rendering_rectangle.x, camera_data.rendering_rectangle.y);
		float2 ip_coords = (make_float2(real_pixel) + jitter) / make_float2(camera_data.camera_size) * 2.0f - 1.0f;
		float3 a_coords = make_float3(ip_coords, 1.0f);
		float3 vec = camera_data.inv_calibration_matrix * a_coords;
		float3 origin = camera_data.eye;
		float3 direction = normalize(vec);
		Ray ray(origin, direction, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
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
}

RT_PROGRAM void empty() {}