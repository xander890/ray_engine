#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"

int ReferenceBSSRDF::entry_point = -1;
int ReferenceBSSRDF::entry_point_post = -1;

void ReferenceBSSRDF::load_data()
{
	context["resulting_flux_intermediate"]->setBuffer(mBSSRDFBufferIntermediate);
	context["resulting_flux"]->setBuffer(mBSSRDFBuffer);
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["ref_frame_number"]->setUint(mRenderedFrames);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);
	context["reference_bssrdf_theta_i"]->setFloat(mThetai);
	context["reference_bssrdf_theta_s"]->setFloat(mThetas);
	context["reference_bssrdf_radius"]->setFloat(mRadius);
	context["reference_bssrdf_g"]->setFloat(mAsymmetry);
	context["reference_bssrdf_albedo"]->setFloat(mAlbedo);
	context["reference_bssrdf_extinction"]->setFloat(mExtinction);
	context["reference_bssrdf_rel_ior"]->setFloat(mIor);
}

void ReferenceBSSRDF::set_material_parameters(float albedo, float extinction, float g, float eta)
{
	mAlbedo = albedo;
	mExtinction = extinction;
	mAsymmetry = g;
	mIor = eta;
}

void ReferenceBSSRDF::set_samples(int samples)
{
}

void ReferenceBSSRDF::set_max_iterations(int max_iter)
{
}

void ReferenceBSSRDF::reset()
{
	float* buff = reinterpret_cast<float*>(mBSSRDFBuffer->map());
	memset(buff, 0, mHemisphereSize.x*mHemisphereSize.y * sizeof(float));
	mBSSRDFBuffer->unmap();
	mRenderedFrames = 0;
}

void ReferenceBSSRDF::init()
{
	std::string ptx_path = get_path_ptx("reference_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if(entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	mBSSRDFBufferIntermediate = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	mBSSRDFBufferIntermediate->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBufferIntermediate->setSize(mHemisphereSize.x, mHemisphereSize.y);

	mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	mBSSRDFBuffer->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);
}

void ReferenceBSSRDF::render()
{
	context->launch(entry_point, mSamples);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	mRenderedFrames++;
}

void ReferenceBSSRDF::on_draw(bool show_material_params)
{
	bool changed = false;
	changed |= ImmediateGUIDraw::SliderFloat("Incoming light angle (deg.)", &mThetai, 0, 90);
	changed |= ImmediateGUIDraw::InputFloat("Radius", &mRadius);
	changed |= ImmediateGUIDraw::SliderFloat("Angle on plane", &mThetas, 0, 360);
	if (show_material_params)
	{
		changed |= ImmediateGUIDraw::InputFloat("Albedo##RefAlbedo", &mAlbedo);
		changed |= ImmediateGUIDraw::InputFloat("Extinction##RefExtinction", &mExtinction);
		changed |= ImmediateGUIDraw::InputFloat("G##RefAsymmetry", &mAsymmetry);
		changed |= ImmediateGUIDraw::InputFloat("Relative IOR##RefRelIOR", &mIor);
	}

	changed |= ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples);
	changed |= ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations);
	if (changed)
		reset();
}

int HemisphereBSSRDFShader::entry_point_output = -1;

HemisphereBSSRDFShader::HemisphereBSSRDFShader(HemisphereBSSRDFShader & other) : Shader(ShaderInfo(other.illum, other.shader_path, other.shader_name))
{
	mCameraWidth = other.mCameraWidth;
	mCameraHeight = other.mCameraHeight;
	ref_impl = nullptr;
	initialize_shader(other.context);
}

void HemisphereBSSRDFShader::init_output()
{
	std::string ptx_path_output = get_path_ptx("render_bssrdf_hemisphere.cu");
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	if(entry_point_output == -1)
		entry_point_output = add_entry_point(context, ray_gen_program_output);

	mBSSRDFBufferTexture = context->createBuffer(RT_BUFFER_INPUT);
	mBSSRDFBufferTexture->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBufferTexture->setSize(ref_impl->get_hemisphere_size().x, ref_impl->get_hemisphere_size().y);

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

void HemisphereBSSRDFShader::initialize_shader(optix::Context ctx)
{
	 Shader::initialize_shader(ctx);
	 //in static constructor

	 if (ref_impl == nullptr)
	 {
		 ref_impl = std::make_unique<ReferenceBSSRDF>(context);
		 ref_impl->init();
	 }

	 init_output();

	 reset();

}

void HemisphereBSSRDFShader::initialize_mesh(Mesh& object)
{

}

void HemisphereBSSRDFShader::pre_trace_mesh(Mesh& object)
{	
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(0, -1, -1);
	ref_impl->render();
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	void* source = ref_impl->get_output_buffer()->map();
	void* dest = mBSSRDFBufferTexture->map();
	memcpy(dest, source, ref_impl->get_hemisphere_size().x*ref_impl->get_hemisphere_size().y * sizeof(float));
	ref_impl->get_output_buffer()->unmap();
	mBSSRDFBufferTexture->unmap();
}

void HemisphereBSSRDFShader::post_trace_mesh(Mesh & object)
{
	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool HemisphereBSSRDFShader::on_draw()
{
	
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);

	ImmediateGUIDraw::Checkbox("Use parameters from material 0", &mUseMeshParameters);

	ref_impl->on_draw(!mUseMeshParameters);

	return false;
}

void HemisphereBSSRDFShader::load_data(Mesh & object)
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
