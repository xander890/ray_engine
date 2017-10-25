#include "bssrdf_visualizer.h"
#include "immediate_gui.h"
#include "optix_utils.h"


int BSSRDFPlaneRenderer::entry_point_output = -1;



void BSSRDFPlaneRenderer::reset()
{
}

BSSRDFPlaneRenderer::BSSRDFPlaneRenderer(const ShaderInfo & shader_info, int camera_width, int camera_height) : Shader(shader_info),
mCameraWidth(camera_width),
mCameraHeight(camera_height)
{

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

	reset();

}

void BSSRDFPlaneRenderer::initialize_mesh(Mesh& object)
{

}

void BSSRDFPlaneRenderer::pre_trace_mesh(Mesh& object)
{
	
}

void BSSRDFPlaneRenderer::post_trace_mesh(Mesh & object)
{
	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool BSSRDFPlaneRenderer::on_draw()
{
	ImmediateGUIDraw::SliderInt("Angle of incoming light", &mAngle, -89, 89);
	ImmediateGUIDraw::InputFloat("Scalar multiplier", &mMult);
	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);
	ImmediateGUIDraw::Combo("Channel to render", (int*)&mChannel, "R\0G\0B", 3);
	return false;
}

void BSSRDFPlaneRenderer::load_data(Mesh & object)
{
	context["scale_multiplier"]->setFloat(mMult);
	context["show_false_colors"]->setUint(mShowFalseColors);
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mParameters->map());
	*cc = object.get_main_material()->get_data().scattering_properties;
	mParameters->unmap();
	context["reference_bssrdf_theta_i"]->setFloat(deg2rad(static_cast<float>(mAngle)));
	context["reference_bssrdf_rel_ior"]->setFloat(object.get_main_material()->get_data().relative_ior);		
	context["channel_to_show"]->setUint(mChannel);
}
