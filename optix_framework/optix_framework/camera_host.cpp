#include "camera_host.h"
#include <optix_world.h>
#include <algorithm>
#include <camera.h>
#include "folders.h"
#include "shader_factory.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include <memory>

using namespace optix;
using namespace std;

Camera::Camera(optix::Context & context, PinholeCameraType::EnumType camera_type, unsigned int width, unsigned int height, unsigned int downsampling, optix::int4 rendering_rect)
{
    const string ptx_path = get_path_ptx("pinhole_camera.cu");
    string camera_name = (camera_type == PinholeCameraType::INVERSE_CAMERA_MATRIX) ? "pinhole_camera_w_matrix" : "pinhole_camera";
    Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, camera_name);
	entry_point = add_entry_point(context, ray_gen_program);
    context->setExceptionProgram(entry_point, context->createProgramFromPTXFile(ptx_path, "exception"));

    data = std::make_unique<CameraData>();
	data->camera_size = make_uint2(width, height);
	if (std::any_of(&rendering_rect.x, &rendering_rect.w, [](int & v){ return v == -1; }))
	{
		data->rendering_rectangle = make_uint4(0, 0, width, height);
	}
	else
	{
		data->rendering_rectangle = make_uint4(rendering_rect.x, rendering_rect.y, rendering_rect.z, rendering_rect.w);
	}
	data->downsampling = downsampling;
	data->rendering_rectangle.z /= downsampling;
	data->rendering_rectangle.w /= downsampling;
	data->U = make_float3(0);
	data->V = make_float3(0);
	data->W = make_float3(0);
	data->eye = make_float3(0);
	data->inv_calibration_matrix = optix::Matrix3x3::identity();
	data->render_bounds = make_uint4(0, 0, width, height);
}

void Camera::update_camera(const SampleScene::RayGenCameraData& camera_data)
{
	data->W = camera_data.W;
	data->U = camera_data.U;
	data->V = camera_data.V;
	data->eye = camera_data.eye;
}

void Camera::set_into_gpu(optix::Context & context) { 
    context["camera_data"]->setUserData(sizeof(CameraData), data.get()); 
}

bool Camera::on_draw()
{
	bool changed = false;
	if (ImmediateGUIDraw::CollapsingHeader("Camera"))
	{
		ImmediateGUIDraw::InputFloat3("Camera eye", (float*)&data->eye, -1, ImGuiInputTextFlags_ReadOnly);
		ImmediateGUIDraw::InputFloat3("Camera U", (float*)&data->U, -1, ImGuiInputTextFlags_ReadOnly);
		ImmediateGUIDraw::InputFloat3("Camera V", (float*)&data->V, -1, ImGuiInputTextFlags_ReadOnly);
		ImmediateGUIDraw::InputFloat3("Camera W", (float*)&data->W, -1, ImGuiInputTextFlags_ReadOnly);
		changed |= ImmediateGUIDraw::InputInt4("Render Bounds", (int*)&data->render_bounds);
	}
	return changed;
}

unsigned int Camera::get_width() const { return data->render_bounds.z; }
unsigned int Camera::get_height() const { return data->render_bounds.w; }

int Camera::get_id() const
{
    return entry_point;
}
