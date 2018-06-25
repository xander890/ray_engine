#include "camera_host.h"
#include <optix_world.h>
#include <algorithm>
#include <camera_common.h>
#include "folders.h"
#include "shader_factory.h"
#include "immediate_gui.h"
#include "optix_host_utils.h"
#include "math_utils.h"
#include <memory>
#include <Eigen/Dense>

#define ISFINITE std::isfinite

/*
  Assigns src to dst if
  src is not inf and nan!

  dst = isReal(src) ? src : dst;
*/

float assignWithCheck( float& dst, const float &src )
{
	if( ISFINITE( src ) ) {
		dst = src;
	}

	return dst;
}

/*
  Assigns src to dst if all src
  components are neither inf nor nan!

  dst = isReal(src) ? src : dst;
*/

float3 assignWithCheck( float3& dst, const float3 &src )
{
	if( ISFINITE( src.x ) && ISFINITE( src.y ) && ISFINITE( src.z ) ) {
		dst = src;
	}

	return dst;
}

optix::Matrix4x4 initWithBasis( const float3& u,
		const float3& v,
		const float3& w,
		const float3& t )
{
	float m[16];
	m[0] = u.x;
	m[1] = v.x;
	m[2] = w.x;
	m[3] = t.x;

	m[4] = u.y;
	m[5] = v.y;
	m[6] = w.y;
	m[7] = t.y;

	m[8] = u.z;
	m[9] = v.z;
	m[10] = w.z;
	m[11] = t.z;

	m[12] = 0.0f;
	m[13] = 0.0f;
	m[14] = 0.0f;
	m[15] = 1.0f;

	return optix::Matrix4x4( m );
}

optix::Matrix3x3 initWithBasis(const float3& u,
	const float3& v,
	const float3& w)
{
	float m[9];
	m[0] = u.x;
	m[1] = v.x;
	m[2] = w.x;

	m[3] = u.y;
	m[4] = v.y;
	m[5] = w.y;

	m[6] = u.z;
	m[7] = v.z;
	m[8] = w.z;

	return optix::Matrix3x3(m);
}

namespace Eigen
{
	Matrix3f optix_to_eigen(optix::Matrix3x3 & mat)
	{
		Matrix3f ret = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(mat.getData());
		return ret;
	}

	optix::Matrix3x3 eigen_to_optix(Matrix3f & mat)
	{
		optix::Matrix3x3 ret;
		Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(ret.getData(), mat.rows(), mat.cols()) = mat;
		return ret;
	}
}
bool Camera::on_draw()
{
	bool changed = false;
    static char buf[64] = "";
    strcpy(buf, mName.c_str());
    if(ImmediateGUIDraw::InputText("Name", buf, 64))
    {
        mName = std::string(buf);
    }
    changed |= ImmediateGUIDraw::InputFloat3("Camera mEye", &mEye.x, -1);
    changed |= ImmediateGUIDraw::InputFloat3("Look at", &mLookAt.x, -1);
    changed |= ImmediateGUIDraw::InputFloat3("Up", &mUp.x, -1);

    if(changed)
        setup();

	return changed;
}


int Camera::get_id() const
{
    return mEntryPoint;
}

void Camera::init()
{
	const std::string ptx_path = get_path_ptx("pinhole_camera.cu");
	std::string camera_name = "pinhole_camera";
	optix::Program ray_gen_program = mContext->createProgramFromPTXFile(ptx_path, camera_name);
	mEntryPoint = add_entry_point(mContext, ray_gen_program);
	mContext->setExceptionProgram(mEntryPoint, mContext->createProgramFromPTXFile(ptx_path, "exception"));

	mData = std::make_unique<CameraData>();
    mName = "camera_" + std::to_string(mEntryPoint);
}


void Camera::set_aspect_ratio(float ratio)
{
	float realRatio = ratio;

	const float* inputAngle = 0;
	float* outputAngle = 0;
	switch(mParameters->ratio_mode) {
	case AspectRatioMode::KEEP_HORIZONTAL:
			inputAngle = &mParameters->hfov;
			outputAngle = &mParameters->vfov;
			realRatio = 1.f/ratio;
			break;
	case AspectRatioMode::KEEP_VERTICAL:
			inputAngle = &mParameters->vfov;
			outputAngle = &mParameters->hfov;
			break;
	case AspectRatioMode::KEEP_NONE:
			return;
			break;
	}

	*outputAngle = rad2deg(2.0f*atanf(realRatio*tanf(deg2rad(0.5f*(*inputAngle)))));

	setup();
}


void Camera::setup()
{
    mData->eye = mEye;
	mData->W = assignWithCheck( mData->W, mLookAt - mEye );  // do not normalize data->W -- implies focal length
	float W_len = length( mData->W );
	mUp = assignWithCheck( mUp, normalize(mUp));
	mData->U = assignWithCheck( mData->U, normalize( cross(mData->W, mUp) ) );
	mData->V = assignWithCheck( mData->V, normalize( cross(mData->U, mData->W) ) );
	float ulen = W_len * tanf(deg2rad(mParameters->hfov*0.5f));
	mData->U = assignWithCheck( mData->U, mData->U * ulen );
	float vlen = W_len * tanf(deg2rad(mParameters->vfov*0.5f));
	mData->V = assignWithCheck( mData->V, mData->V * vlen );
	mContext["camera_data"]->setUserData(sizeof(CameraData), mData.get());
}

void Camera::get_eye_uvw(float3& eye_out, float3& U, float3& V, float3& W)
{
	eye_out = mEye;
	U = mData->U;
	V = mData->V;
	W = mData->W;
}

void Camera::scale_fov(float scale)
{
	const float fov_min = 0.0f;
	const float fov_max = 120.0f;
	float hfov_new = rad2deg(2*atanf(scale*tanf(deg2rad(mParameters->hfov*0.5f))));
	hfov_new = optix::clamp(hfov_new, fov_min, fov_max);
	float vfov_new = rad2deg(2*atanf(scale*tanf(deg2rad(mParameters->vfov*0.5f))));
	vfov_new = optix::clamp(vfov_new, fov_min, fov_max);

	mParameters->hfov = assignWithCheck( mParameters->hfov, hfov_new );
	mParameters->vfov = assignWithCheck( mParameters->vfov, vfov_new );

	setup();
}

void Camera::translate(float2 t)
{
	float3 trans = mData->U*t.x + mData->V*t.y;

	mEye = assignWithCheck( mEye, mEye + trans );
	mLookAt = assignWithCheck( mLookAt, mLookAt + trans );

	setup();
}


// Here scale will move the mEye point closer or farther away from the
// mLookAt point.  If you want an invertable value feed it
// (previous_scale/(previous_scale-1)
void Camera::dolly(float scale)
{
	// Better make sure the scale isn't exactly one.
	if (scale == 1.0f) return;
	float3 d = (mLookAt - mEye) * scale;
	mEye  = assignWithCheck( mEye, mEye + d );

	setup();
}

void Camera::transform( const optix::Matrix4x4& trans )
{
	float3 cen = mLookAt;         // TODO: Add logic for various rotation types (eg, flythrough)

	optix::Matrix4x4 frame = initWithBasis( normalize(mData->U),
			normalize(mData->V),
			normalize(-mData->W),
			cen );
	optix::Matrix4x4 frame_inv = frame.inverse();

	optix::Matrix4x4 final_trans = frame * trans * frame_inv;
	optix::float4 up4     = make_float4( mUp );
	optix::float4 eye4    = make_float4( mEye );
	eye4.w         = 1.0f;
	optix::float4 lookat4 = make_float4( mLookAt );
	lookat4.w      = 1.0f;


	mUp     = assignWithCheck( mUp, make_float3( final_trans*up4 ) );
	mEye    = assignWithCheck( mEye, make_float3( final_trans*eye4 ) );
	mLookAt = assignWithCheck( mLookAt, make_float3( final_trans*lookat4 ) );

	setup();
}

Camera::Camera(optix::Context &context, CameraParameters parameters, const std::string& mname, float3 eye, float3 lookat, float3 up):
        mEye(eye),
		mLookAt(lookat)
		, mUp(up)
		, mContext(context)
{
	init();
    if(mname != "")
        mName = mname;
	set_parameters(parameters);
    setup();
}

void Camera::set_parameters(CameraParameters param)
{
	mParameters = std::make_unique<CameraParameters>(param);
	mData->downsampling = mParameters->downsampling;
	float3 cen = mLookAt;
    mData->view_matrix = initWithBasis( normalize(mData->U), normalize(mData->V), normalize(-mData->W));
	Eigen::Matrix3f e = Eigen::optix_to_eigen(mData->view_matrix);
	e = e.inverse().eval();
	mData->inv_view_matrix = Eigen::eigen_to_optix(e);
}

void Camera::set_eye_lookat_up(float3 eye_in, float3 lookat_in, float3 up_in)
{
    mEye = eye_in;
    mLookAt = lookat_in;
    mUp = up_in;
    setup();
}

void Camera::set_as_other_camera(std::shared_ptr<Camera> &camera)
{
	set_parameters(*camera->mParameters);
	mEye = camera->mEye;
	mLookAt = camera->mLookAt;
	mUp = camera->mUp;
    mName = camera->mName;
    mEntryPoint = camera->mEntryPoint;
	setup();
}
