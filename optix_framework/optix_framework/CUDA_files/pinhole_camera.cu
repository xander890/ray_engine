// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011


#include <device_common_data.h>
#include <random.h>
#include <environment_map.h>

using namespace optix;

// Camera variables
rtDeclareVariable(float3,   eye, , );
rtDeclareVariable(float3,   U, , );
rtDeclareVariable(float3,   V, , );
rtDeclareVariable(float3,   W, , );

// Ray generation variables
rtDeclareVariable(uint,     frame, , );

// Window variables
rtBuffer<float4, 2> output_buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(uint2, debug_index, , );
rtDeclareVariable(int4, render_bounds, , );
rtDeclareVariable(uint4, rendering_rectangle, , );
rtDeclareVariable(uint2, camera_size, , );
rtDeclareVariable(uint, downsampling, , );
// Exception and debugging variables
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(Matrix3x3, inv_calibration_matrix, , ) = Matrix3x3();

rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

//#define TIME_VIEW

__forceinline__ __device__ bool check_bounds()
{
	return	launch_index.x >= render_bounds.x && launch_index.x < render_bounds.x + render_bounds.z &&
		launch_index.y >= render_bounds.y && launch_index.y < render_bounds.y + render_bounds.w;
}

__forceinline__ __device__ void trace(const Ray& ray, PerRayData_radiance & prd)
{
	rtTrace(top_object, ray, prd);

	if (isfinite(prd.result.x) && isfinite(prd.result.y) && isfinite(prd.result.z))
	{
		float4 curr_sum = (frame != 0) ? output_buffer[launch_index] * ((float)frame) : make_float4(0.0f);
		output_buffer[launch_index] = (make_float4(prd.result, 0.0f) + curr_sum) / ((float)(frame + 1));
		optix_print("Final color: %f %f %f, sum = %f %f %f\n", prd.result.x, prd.result.y, prd.result.z, output_buffer[launch_index].x, output_buffer[launch_index].y, output_buffer[launch_index].z);

		float deblen = length(make_float2(debug_index - launch_index));
		if (debug_index.x == launch_index.x && debug_index.y == launch_index.y)
		{
			output_buffer[launch_index] = make_float4(1, 0, 0, 1);
		}
	}
}

__forceinline__ __device__ void init_payload(PerRayData_radiance & prd)
{
	prd.importance = 1.0f;
	prd.depth = 0;
	prd.seed = tea<16>(launch_dim.x*launch_index.y + launch_index.x, frame);
	prd.flags = 0;
	prd.flags &= ~(RayFlags::HIT_DIFFUSE_SURFACE); //Just for clarity
	prd.flags |= RayFlags::USE_EMISSION;
	prd.colorband = -1;
	prd.result = make_float3(0);
}

RT_PROGRAM void pinhole_camera()
{
	if (check_bounds())
	{

	#ifdef TIME_VIEW
		clock_t t0 = clock(); 
	#endif
		PerRayData_radiance prd;
		init_payload(prd);
		float2 jitter = make_float2(rnd(prd.seed), rnd(prd.seed)) * downsampling;
		uint2 real_pixel = launch_index * downsampling + make_uint2(rendering_rectangle.x, rendering_rectangle.y);
		float2 ip_coords = (make_float2(real_pixel) + jitter) / make_float2(camera_size) * 2.0f - 1.0f;

	#ifdef ORTHO
		float3 direction = normalize(W);
		float3 origin = eye + ip_coords.x*U + ip_coords.y*V;
	#else
		float3 origin = eye;
		float3 direction = normalize(ip_coords.x*U + ip_coords.y*V + W);
	#endif
		Ray ray(origin, direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
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

#ifdef TIME_VIEW
		clock_t t0 = clock();
#endif
		PerRayData_radiance prd;
		init_payload(prd);

		float2 jitter = make_float2(rnd(prd.seed), rnd(prd.seed))* downsampling;
		uint2 real_pixel = launch_index * downsampling + make_uint2(rendering_rectangle.x, rendering_rectangle.y);
		float2 ip_coords = (make_float2(real_pixel) + jitter) / make_float2(camera_size) * 2.0f - 1.0f;
		float3 a_coords = make_float3(ip_coords, 1.0f);
		float3 vec = inv_calibration_matrix * a_coords;
		float3 origin = eye;
		float3 direction = normalize(vec);
		Ray ray(origin, direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
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