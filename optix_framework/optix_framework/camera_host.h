#pragma once

#include "SampleScene.h"
#include <memory>
#include <optix_world.h>
#include "enums.h"
#include "camera.h"

class Camera
{
public:
	virtual ~Camera() = default;
	std::unique_ptr<CameraData> data;

    Camera(optix::Context & context, PinholeCameraType::EnumType camera_type, unsigned int width, unsigned int height, unsigned int downsampling = 1, optix::int4 rendering_rect = optix::make_int4(-1));
	virtual void update_camera(const SampleScene::RayGenCameraData & camera_data);
    virtual void set_into_gpu(optix::Context & context);

	virtual bool on_draw();

	unsigned int get_width() const;
	unsigned int get_height() const;
	int get_id() const;

	unsigned int get_entry_point() const { return entry_point; }

private:	
	unsigned int entry_point;
};

