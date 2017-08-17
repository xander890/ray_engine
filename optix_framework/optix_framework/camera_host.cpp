#include "camera_host.h"
#include <optix_world.h>
#include <algorithm>
#include <camera.h>
#include "folders.h"
#include "shader_factory.h"

using namespace optix;
using namespace std;

Camera::Camera(optix::Context & context, PinholeCameraDefinitionType::EnumType camera_type, int width, int height, int downsampling, optix::int4 rendering_rect)
{
    context->setEntryPointCount(as_integer(CameraType::COUNT));
    const string ptx_path = get_path_ptx("pinhole_camera.cu");
    string camera_name = (camera_type == PinholeCameraDefinitionType::INVERSE_CAMERA_MATRIX) ? "pinhole_camera_w_matrix" : "pinhole_camera";
    Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, camera_name);
    context->setRayGenerationProgram(as_integer(CameraType::STANDARD_RT), ray_gen_program);
    context->setExceptionProgram(as_integer(CameraType::STANDARD_RT), context->createProgramFromPTXFile(ptx_path, "exception"));

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

void Camera::set_into_gui(GUI * gui) {
    const char* camera_g = "Camera";
    gui->addDirVariable("Camera eye", &data->eye, camera_g);
    gui->setReadOnly("Camera eye");
    gui->addDirVariable("Camera U", &data->U, camera_g);
    gui->setReadOnly("Camera U");
    gui->addDirVariable("Camera V", &data->V, camera_g);
    gui->setReadOnly("Camera V");
    gui->addDirVariable("Camera W", &data->W, camera_g);
    gui->setReadOnly("Camera W");
    gui->addIntVariable("X", (int*)&data->render_bounds.x, camera_g, 0, data->camera_size.x);
    gui->addIntVariable("Y", (int*)&data->render_bounds.y, camera_g, 0, data->camera_size.y);
    gui->addIntVariable("W", (int*)&data->render_bounds.z, camera_g, 0, data->camera_size.x);
    gui->addIntVariable("H", (int*)&data->render_bounds.w, camera_g, 0, data->camera_size.y);

}

int Camera::get_width() const { return data->render_bounds.z; }
int Camera::get_height() const { return data->render_bounds.w; }