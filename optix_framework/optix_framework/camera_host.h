#pragma once
#include "gui.h"
#include "SampleScene.h"
#include <memory>
#include <optix_world.h>

struct CameraData;

class Camera
{
public:
	virtual ~Camera() = default;
	std::unique_ptr<CameraData> data;

	Camera(int width, int height, int downsampling = 1, optix::int4 rendering_rect = optix::make_int4(-1));
	virtual void update_camera(const SampleScene::RayGenCameraData & camera_data);
    virtual void set_into_gpu(optix::Context & context);
	//virtual void set_into_gui(GUI * gui) = 0;

    int get_width() const;
    int get_height() const;
};

