#include "bssrdf_visualizer.h"
#include "immediate_gui.h"
#include "optix_utils.h"


int BSSRDFVisualizer::entry_point_output = -1;



void BSSRDFVisualizer::reset()
{
}

BSSRDFVisualizer::BSSRDFVisualizer(const ShaderInfo & shader_info, int camera_width, int camera_height) : Shader(shader_info),
mCameraWidth(camera_width),
mCameraHeight(camera_height)
{

}

void BSSRDFVisualizer::initialize_shader(optix::Context ctx)
{
	Shader::initialize_shader(ctx);
	//in static constructor

	std::string ptx_path_output = get_path_ptx("bssrdf_plane_visualizer.cu");
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render");

	if (entry_point_output == -1)
		entry_point_output = add_entry_point(context, ray_gen_program_output);

	mParameters = create_buffer<ScatteringMaterialProperties>(context, 1);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mParameters->getId());
	context["planar_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);

	reset();

}

void BSSRDFVisualizer::initialize_mesh(Mesh& object)
{

}

void BSSRDFVisualizer::pre_trace_mesh(Mesh& object)
{
	
}

void BSSRDFVisualizer::post_trace_mesh(Mesh & object)
{
	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool BSSRDFVisualizer::on_draw()
{
	ImmediateGUIDraw::SliderInt("Angle of incoming light", &mAngle, -89, 89);
	ImmediateGUIDraw::InputFloat("Scalar multiplier", &mMult);
	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);
	ImmediateGUIDraw::Combo("Channel to render", (int*)&mChannel, "R\0G\0B", 3);
	return false;
}

void BSSRDFVisualizer::load_data(Mesh & object)
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
