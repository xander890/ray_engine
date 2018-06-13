#include "bssrdf_visualizer.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "object_host.h"

int BSSRDFPlaneRenderer::entry_point_output = -1;



void BSSRDFPlaneRenderer::reset()
{
}

BSSRDFPlaneRenderer::BSSRDFPlaneRenderer(const ShaderInfo & shader_info, int camera_width, int camera_height) : Shader(shader_info),
mCameraWidth(camera_width),
mCameraHeight(camera_height)
{

}

BSSRDFPlaneRenderer::BSSRDFPlaneRenderer(BSSRDFPlaneRenderer & copy) : Shader(copy)
{
	auto type = copy.mBSSRDF->get_type();
	mBSSRDF = BSSRDF::create(context, type);

	mCameraWidth = copy.mCameraWidth;
	mCameraHeight = copy.mCameraHeight;

	// Gui
	mShowFalseColors = copy.mShowFalseColors;
	mAngle = copy.mAngle;
	mMult = copy.mMult;
	mChannel = copy.mChannel;

	mParameters = clone_buffer(copy.mParameters, RT_BUFFER_INPUT); 
	
}

void BSSRDFPlaneRenderer::initialize_shader(optix::Context ctx)
{
	Shader::initialize_shader(ctx);
	//in static constructor

	std::string ptx_path_output = get_path_ptx("bssrdf_plane_visualizer.cu");
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render");

	if (entry_point_output == -1)
		entry_point_output = add_entry_point(context, ray_gen_program_output);

	mParameters = create_buffer<ScatteringMaterialProperties>(context, RT_BUFFER_INPUT, 1);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mParameters->getId());
	context["planar_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);

	mBSSRDF = BSSRDF::create(context, ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF);

	reset();

}

void BSSRDFPlaneRenderer::initialize_material(MaterialHost &object)
{
	ScatteringDipole::Type t = mBSSRDF->get_type();
	object.get_optix_material()["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &t);
}

void BSSRDFPlaneRenderer::pre_trace_mesh(Object& object)
{
	
}

void BSSRDFPlaneRenderer::post_trace_mesh(Object & object)
{
	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool BSSRDFPlaneRenderer::on_draw()
{
	mBSSRDF->on_draw();
	ImmediateGUIDraw::SliderInt("Angle of incoming light", &mAngle, -89, 89);
	ImmediateGUIDraw::InputFloat("Scalar multiplier", &mMult);
	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);
	ImmediateGUIDraw::Combo("Channel to render", (int*)&mChannel, "R\0G\0B", 3);
	return false;
}

void BSSRDFPlaneRenderer::load_data(MaterialHost &mat)
{
	context["scale_multiplier"]->setFloat(mMult);
	context["show_false_colors"]->setUint(mShowFalseColors);
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mParameters->map());
	*cc = mat.get_data().scattering_properties;
	mParameters->unmap();
	context["reference_bssrdf_theta_i"]->setFloat(deg2rad(static_cast<float>(mAngle)));
	context["reference_bssrdf_rel_ior"]->setFloat(dot(mat.get_data().index_of_refraction, optix::make_float3(1)) / 3);
    mBSSRDF->load(mat.get_data().index_of_refraction, mat.get_data().scattering_properties);
	ScatteringDipole::Type t = mBSSRDF->get_type();
	mat.get_optix_material()["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &t);
	context["channel_to_show"]->setUint(mChannel);
}
