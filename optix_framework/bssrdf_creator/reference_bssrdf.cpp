#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_host_utils.h"
#include "GL/glew.h"
#include <bssrdf_creator.h>
#include "object_host.h"

int HemisphereBSSRDFShader::entry_point_output = -1;

HemisphereBSSRDFShader::HemisphereBSSRDFShader(HemisphereBSSRDFShader & other) : Shader(ShaderInfo(other.illum, other.shader_path, other.shader_name))
{
	mCameraWidth = other.mCameraWidth;
	mCameraHeight = other.mCameraHeight;
	ref_impl = other.ref_impl;
	initialize_shader(other.context);
}

void HemisphereBSSRDFShader::init_output()
{
	std::string ptx_path_output = get_path_ptx("render_bssrdf.cu");
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	if(entry_point_output == -1)
		entry_point_output = add_entry_point(context, ray_gen_program_output);

	mBSSRDFBufferTexture = context->createBuffer(RT_BUFFER_INPUT);
	mBSSRDFBufferTexture->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBufferTexture->setSize(ref_impl->get_size().x, ref_impl->get_size().y);

	mBSSRDFHemisphereTex = context->createTextureSampler();
	mBSSRDFHemisphereTex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	mBSSRDFHemisphereTex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	mBSSRDFHemisphereTex->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
	mBSSRDFHemisphereTex->setBuffer(mBSSRDFBufferTexture);
	mBSSRDFHemisphereTex->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
	mBSSRDFHemisphereTex->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
}

void HemisphereBSSRDFShader::reset()
{
	ref_impl->reset();
}

HemisphereBSSRDFShader::HemisphereBSSRDFShader(const ShaderInfo & shader_info, std::unique_ptr<BSSRDFRenderer>& creator, int camera_width, int camera_height) : Shader(shader_info),
mCameraWidth(camera_width),
mCameraHeight(camera_height)
{
	if (creator != nullptr)
		ref_impl = std::move(creator);
	else
		ref_impl = std::make_unique<BSSRDFRendererSimulated>(context);
}

void HemisphereBSSRDFShader::initialize_shader(optix::Context ctx)
{
	 Shader::initialize_shader(ctx);
	 //in static constructor

	 if (ref_impl == nullptr)
	 {
		 ref_impl = std::make_unique<BSSRDFRendererSimulated>(context);

	 }

	 init_output();

	 reset();

}

void HemisphereBSSRDFShader::initialize_mesh(Object& object)
{

}

void HemisphereBSSRDFShader::pre_trace_mesh(Object& object)
{	
	ref_impl->render();

	void* source = ref_impl->get_output_buffer()->map();
	void* dest = mBSSRDFBufferTexture->map();
	memcpy(dest, source, ref_impl->get_size().x*ref_impl->get_size().y * sizeof(float));
	ref_impl->get_output_buffer()->unmap();
	mBSSRDFBufferTexture->unmap();
}

void HemisphereBSSRDFShader::post_trace_mesh(Object & object)
{
	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool HemisphereBSSRDFShader::on_draw()
{
	
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);

	ImmediateGUIDraw::Checkbox("Use parameters from material 0", &mUseMeshParameters);

	ref_impl->on_draw(!mUseMeshParameters? BSSRDFRenderer::SHOW_ALL : BSSRDFRenderer::HIDE_ALL);

	return false;
}

void HemisphereBSSRDFShader::load_data(Object & object)
{
	int s = mBSSRDFHemisphereTex->getId();
	context["resulting_flux_tex"]->setUserData(sizeof(TexPtr),&(s));
	context["show_false_colors"]->setUint(mShowFalseColors);
	context["reference_scale_multiplier"]->setFloat(mScaleMultiplier);

	if (mUseMeshParameters)
	{
		ref_impl->set_material_parameters(object.get_main_material()->get_data().scattering_properties.albedo.x,
			object.get_main_material()->get_data().scattering_properties.extinction.x,
			object.get_main_material()->get_data().scattering_properties.meancosine.x,
			object.get_main_material()->get_data().relative_ior);

	}

	ref_impl->load_data();
}
