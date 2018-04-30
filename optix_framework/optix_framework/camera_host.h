#pragma once

#include "SampleScene.h"
#include <memory>
#include <optix_world.h>
#include "enums.h"
#include "camera.h"
#include "optix_serialize.h"

using optix::float2;
using optix::float3;

enum AspectRatioMode {
    KeepVertical,
    KeepHorizontal,
    KeepNone
};

struct CameraParameters
{
    float hfov = 60;
    float vfov = 60;
    int width = 512;
    int height = 512;
    int downsampling = 1;
    AspectRatioMode ratio_mode = KeepVertical;
    optix::int4 rendering_rect = optix::make_int4(-1);

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive( CEREAL_NVP(hfov), CEREAL_NVP(vfov), CEREAL_NVP(width), CEREAL_NVP(height), CEREAL_NVP(downsampling), CEREAL_NVP(ratio_mode), CEREAL_NVP(rendering_rect));
    }
};


class Camera
{
public:
	virtual ~Camera() = default;
	std::unique_ptr<CameraData> data;
    std::unique_ptr<CameraParameters> parameters;

    Camera(optix::Context & context, CameraParameters parameters, float3 eye = optix::make_float3(0), float3 lookat= optix::make_float3(1,0,0), float3 up = optix::make_float3(0,1,0));

    void setup();
    void getEyeUVW(float3& eye, float3& U, float3& V, float3& W);
    void scaleFOV(float);
    void translate(float2);
    void dolly(float);
    void transform( const optix::Matrix4x4& trans );
    void setAspectRatio(float ratio);
    void setParameters(CameraParameters parameters);
    void setEyeLookatUp(float3 eye, float3 lookat, float3 up);

	virtual bool on_draw();

	unsigned int get_width() const;
	unsigned int get_height() const;
	int get_id() const;

	unsigned int get_entry_point() const { return entry_point; }

private:	
	unsigned int entry_point;
    void init();

    float3 eye, lookat, up;

	friend class cereal::access;
    template<class Archive>
    void load(Archive & archive)
    {
        archive(cereal::make_nvp("parameters", parameters));
        archive(cereal::make_nvp("eye", eye));
        archive(cereal::make_nvp("lookat", lookat));
        archive(cereal::make_nvp("up", up));
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("parameters", parameters));
        archive(cereal::make_nvp("eye", eye));
        archive(cereal::make_nvp("lookat", lookat));
        archive(cereal::make_nvp("up", up));
    }
    optix::Context mContext;
};

