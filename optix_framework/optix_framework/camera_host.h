#pragma once
#include "camera.h"
#include "gui.h"
#include "SampleScene.h"

class Camera
{
public:
	virtual ~Camera() = default;
	CameraData data;

	Camera(int width, int height, int downsampling = 1, optix::int4 rendering_rect = optix::make_int4(-1));
	virtual void update_camera(const SampleScene::RayGenCameraData & camera_data);
	virtual void set_into_gpu(optix::Context & context) { context["camera_data"]->setUserData(sizeof(CameraData), &data); }
	//virtual void set_into_gui(GUI * gui) = 0;

	int get_width() const { return data.render_bounds.z; }
	int get_height() const { return data.render_bounds.w; }
};

