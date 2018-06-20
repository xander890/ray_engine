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
    strcpy(buf, name.c_str());
    if(ImmediateGUIDraw::InputText("Name", buf, 64))
    {
        name = std::string(buf);
    }
    changed |= ImmediateGUIDraw::InputFloat3("Camera eye", &eye.x, -1);
    changed |= ImmediateGUIDraw::InputFloat3("Look at", &lookat.x, -1);
    changed |= ImmediateGUIDraw::InputFloat3("Up", &up.x, -1);
    changed |= ImmediateGUIDraw::InputInt4("Render Bounds", (int*)&data->render_bounds.x);

    if(changed)
        setup();

	return changed;
}

unsigned int Camera::get_width() const { return data->render_bounds.z; }
unsigned int Camera::get_height() const { return data->render_bounds.w; }

int Camera::get_id() const
{
    return entry_point;
}

void Camera::init()
{
	const std::string ptx_path = get_path_ptx("pinhole_camera.cu");
	std::string camera_name = "pinhole_camera";
	optix::Program ray_gen_program = mContext->createProgramFromPTXFile(ptx_path, camera_name);
	entry_point = add_entry_point(mContext, ray_gen_program);
	mContext->setExceptionProgram(entry_point, mContext->createProgramFromPTXFile(ptx_path, "exception"));

	data = std::make_unique<CameraData>();
    name = "camera_" + std::to_string(entry_point);
}


void Camera::setAspectRatio(float ratio)
{
	float realRatio = ratio;

	const float* inputAngle = 0;
	float* outputAngle = 0;
	switch(parameters->ratio_mode) {
		case KeepHorizontal:
			inputAngle = &parameters->hfov;
			outputAngle = &parameters->vfov;
			realRatio = 1.f/ratio;
			break;
		case KeepVertical:
			inputAngle = &parameters->vfov;
			outputAngle = &parameters->hfov;
			break;
		case KeepNone:
			return;
			break;
	}

	*outputAngle = rad2deg(2.0f*atanf(realRatio*tanf(deg2rad(0.5f*(*inputAngle)))));

	setup();
}


void Camera::setup()
{
    data->eye = eye;
	data->W = assignWithCheck( data->W, lookat - eye );  // do not normalize data->W -- implies focal length
	float W_len = length( data->W );
	up = assignWithCheck( up, normalize(up));
	data->U = assignWithCheck( data->U, normalize( cross(data->W, up) ) );
	data->V = assignWithCheck( data->V, normalize( cross(data->U, data->W) ) );
	float ulen = W_len * tanf(deg2rad(parameters->hfov*0.5f));
	data->U = assignWithCheck( data->U, data->U * ulen );
	float vlen = W_len * tanf(deg2rad(parameters->vfov*0.5f));
	data->V = assignWithCheck( data->V, data->V * vlen );
	mContext["camera_data"]->setUserData(sizeof(CameraData), data.get());
}

void Camera::getEyeUVW(float3& eye_out, float3& U, float3& V, float3& W)
{
	eye_out = eye;
	U = data->U;
	V = data->V;
	W = data->W;
}

void Camera::scaleFOV(float scale)
{
	const float fov_min = 0.0f;
	const float fov_max = 120.0f;
	float hfov_new = rad2deg(2*atanf(scale*tanf(deg2rad(parameters->hfov*0.5f))));
	hfov_new = optix::clamp(hfov_new, fov_min, fov_max);
	float vfov_new = rad2deg(2*atanf(scale*tanf(deg2rad(parameters->vfov*0.5f))));
	vfov_new = optix::clamp(vfov_new, fov_min, fov_max);

	parameters->hfov = assignWithCheck( parameters->hfov, hfov_new );
	parameters->vfov = assignWithCheck( parameters->vfov, vfov_new );

	setup();
}

void Camera::translate(float2 t)
{
	float3 trans = data->U*t.x + data->V*t.y;

	eye = assignWithCheck( eye, eye + trans );
	lookat = assignWithCheck( lookat, lookat + trans );

	setup();
}


// Here scale will move the eye point closer or farther away from the
// lookat point.  If you want an invertable value feed it
// (previous_scale/(previous_scale-1)
void Camera::dolly(float scale)
{
	// Better make sure the scale isn't exactly one.
	if (scale == 1.0f) return;
	float3 d = (lookat - eye) * scale;
	eye  = assignWithCheck( eye, eye + d );

	setup();
}

void Camera::transform( const optix::Matrix4x4& trans )
{
	float3 cen = lookat;         // TODO: Add logic for various rotation types (eg, flythrough)

	optix::Matrix4x4 frame = initWithBasis( normalize(data->U),
			normalize(data->V),
			normalize(-data->W),
			cen );
	optix::Matrix4x4 frame_inv = frame.inverse();

	optix::Matrix4x4 final_trans = frame * trans * frame_inv;
	optix::float4 up4     = make_float4( up );
	optix::float4 eye4    = make_float4( eye );
	eye4.w         = 1.0f;
	optix::float4 lookat4 = make_float4( lookat );
	lookat4.w      = 1.0f;


	up     = assignWithCheck( up, make_float3( final_trans*up4 ) );
	eye    = assignWithCheck( eye, make_float3( final_trans*eye4 ) );
	lookat = assignWithCheck( lookat, make_float3( final_trans*lookat4 ) );

	setup();
}

Camera::Camera(optix::Context &context, CameraParameters parameters, const std::string& mname, float3 eye, float3 lookat, float3 up):
        eye(eye),
		lookat(lookat)
		, up(up)
		, mContext(context)
{
	init();
    if(mname != "")
        name = mname;
	setParameters(parameters);
    setup();
}

void Camera::setParameters(CameraParameters param)
{
	parameters = std::make_unique<CameraParameters>(param);
	if (std::any_of(&parameters->rendering_rect.x, &parameters->rendering_rect.w, [](int & v){ return v == -1; }))
	{
		data->rendering_rectangle = optix::make_uint4(0, 0, parameters->width, parameters->height);
	}
	else
	{
		data->rendering_rectangle = optix::make_uint4(parameters->rendering_rect.x, parameters->rendering_rect.y, parameters->rendering_rect.z, parameters->rendering_rect.w);
	}
	data->downsampling = parameters->downsampling;
	data->rendering_rectangle.z /= parameters->downsampling;
	data->rendering_rectangle.w /= parameters->downsampling;
	data->render_bounds = optix::make_uint4(0, 0, parameters->width, parameters->height);
	data->camera_size = optix::make_uint2(parameters->width, parameters->height);
	float3 cen = lookat;
    data->view_matrix = initWithBasis( normalize(data->U), normalize(data->V), normalize(-data->W));
	Eigen::Matrix3f e = Eigen::optix_to_eigen(data->view_matrix);
	e = e.inverse().eval();
	data->inv_view_matrix = Eigen::eigen_to_optix(e);
}

void Camera::setEyeLookatUp(float3 eye_in, float3 lookat_in, float3 up_in)
{
    eye = eye_in;
    lookat = lookat_in;
    up = up_in;
    setup();
}

void Camera::setAsOtherCamera(std::shared_ptr<Camera> &camera)
{
	setParameters(*camera->parameters);
	eye = camera->eye;
	lookat = camera->lookat;
	up = camera->up;
    name = camera->name;
    entry_point = camera->entry_point;
	setup();
}
