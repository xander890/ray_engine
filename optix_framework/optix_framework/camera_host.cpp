#include "camera_host.h"
#include <algorithm>

using namespace optix;

Camera::Camera(int width, int height, int downsampling, optix::int4 rendering_rect)
{
	data.camera_size = make_uint2(width, height);
	if (std::any_of(&rendering_rect.x, &rendering_rect.w, [](int & v){ return v == -1; }))
	{
		data.rendering_rectangle = make_uint4(0, 0, width, height);
	}
	else
	{
		data.rendering_rectangle = make_uint4(rendering_rect.x, rendering_rect.y, rendering_rect.z, rendering_rect.w);
	}
	data.downsampling = downsampling;
	data.rendering_rectangle.z /= downsampling;
	data.rendering_rectangle.w /= downsampling;
	data.U = make_float3(0);
	data.V = make_float3(0);
	data.W = make_float3(0);
	data.eye = make_float3(0);
	data.inv_calibration_matrix = optix::Matrix3x3::identity();
	data.render_bounds = make_uint4(0, 0, width, height);
}

void Camera::update_camera(const SampleScene::RayGenCameraData& camera_data)
{
	data.W = camera_data.W;
	data.U = camera_data.U;
	data.V = camera_data.V;
	data.eye = camera_data.eye;
}
