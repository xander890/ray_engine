#pragma once
#include "sample_scene.h"
#include <memory>
#include <optix_world.h>
#include "camera_common.h"
#include "optix_serialize_utils.h"

using optix::float2;
using optix::float3;

#define IMPROVED_ENUM_NAME AspectRatioMode
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(KEEP_VERTICAL,0) ENUMITEM_VALUE(KEEP_HORIZONTAL,1) ENUMITEM_VALUE(KEEP_NONE,2)
#include "improved_enum.inc"

/*
 * Struct to store the camera "intrinsic parameters". At the moment they can model only a pinhole camera.
 */
struct CameraParameters
{
    float hfov = 60;
    float vfov = 60;
    int downsampling = 1;
    AspectRatioMode::Type ratio_mode = AspectRatioMode::KEEP_VERTICAL;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive( CEREAL_NVP(hfov), CEREAL_NVP(vfov), CEREAL_NVP(downsampling), CEREAL_NVP(ratio_mode));
    }
};

/*
 * Main camera class. For now, it models a pinhole camera only.
 */
class Camera
{
public:
	virtual ~Camera() = default;

    Camera(optix::Context & context, CameraParameters parameters, const std::string& name = "", float3 eye = optix::make_float3(0), float3 lookat= optix::make_float3(1,0,0), float3 up = optix::make_float3(0,1,0));

    void setup();
    void get_eye_uvw(float3& eye, float3& U, float3& V, float3& W);
    void scale_fov(float);
    void translate(float2);
    void dolly(float);
    void transform( const optix::Matrix4x4& trans );
    void set_aspect_ratio(float ratio);
    void set_parameters(CameraParameters parameters);
    void set_eye_lookat_up(float3 eye, float3 lookat, float3 up);
    void set_as_other_camera(std::shared_ptr<Camera>& camera);
	virtual bool on_draw();
	int get_id() const;
	unsigned int get_entry_point() const { return mEntryPoint; }

private:
    void init();
	friend class cereal::access;

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("name", mName));
        archive(cereal::make_nvp("parameters", *mParameters));
        archive(cereal::make_nvp("eye", mEye));
        archive(cereal::make_nvp("lookat", mLookAt));
        archive(cereal::make_nvp("up", mUp));
    }

    static void load_and_construct( cereal::XMLInputArchiveOptix & archive, cereal::construct<Camera>& construct )
    {
        std::string name;
        CameraParameters parameters;
        optix::float3 eye,lookat,up;
        archive(cereal::make_nvp("name", name));
        archive(cereal::make_nvp("parameters", parameters));
        archive(cereal::make_nvp("eye", eye));
        archive(cereal::make_nvp("lookat", lookat));
        archive(cereal::make_nvp("up", up));
        optix::Context ctx = archive.get_context();
        construct(ctx, parameters, name, eye, lookat, up);
    }

    optix::Context mContext;
	float3 mEye, mLookAt, mUp;
	std::string mName;
	unsigned int mEntryPoint;
	std::unique_ptr<CameraData> mData;
	std::unique_ptr<CameraParameters> mParameters;
};

