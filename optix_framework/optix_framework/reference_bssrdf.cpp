#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"

inline void BSSRDFCreator::set_geometry_parameters(float theta_i, float theta_s, float r)
{
	mThetai = theta_i;
	mThetas = theta_s;
	mRadius = r;
}

void BSSRDFCreator::load_data()
{
	ScatteringMaterialProperties c;
	fill_scattering_parameters_alternative(c, 1, mIor, make_float3(mAlbedo), make_float3(mExtinction), make_float3(mAsymmetry));
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	*cc = c;
	mProperties->unmap();

	context["resulting_flux_intermediate"]->setBuffer(mBSSRDFBufferIntermediate);
	context["resulting_flux"]->setBuffer(mBSSRDFBuffer);
	context["ref_frame_number"]->setUint(mRenderedFrames);
	context["reference_bssrdf_theta_i"]->setFloat(mThetai);
	context["reference_bssrdf_theta_s"]->setFloat(mThetas);
	context["reference_bssrdf_radius"]->setFloat(mRadius);
	context["reference_bssrdf_rel_ior"]->setFloat(mIor);
}

void BSSRDFCreator::set_material_parameters(float albedo, float extinction, float g, float eta)
{
	//mAlbedo = albedo;
	//mExtinction = extinction;
	//mAsymmetry = g;
	//mIor = eta;
}

void BSSRDFCreator::reset()
{
	clear_buffer(mBSSRDFBuffer);
	clear_buffer(mBSSRDFBufferIntermediate);
	mRenderedFrames = 0;
	
}

void BSSRDFCreator::init()
{
	if (mBSSRDFBufferIntermediate.get() == nullptr)
	{
		mBSSRDFBufferIntermediate = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
		mBSSRDFBufferIntermediate->setFormat(RT_FORMAT_FLOAT);
		mBSSRDFBufferIntermediate->setSize(mHemisphereSize.x, mHemisphereSize.y);
	}

	if (mBSSRDFBuffer.get() == nullptr)
	{
		mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
		mBSSRDFBuffer->setFormat(RT_FORMAT_FLOAT);
		mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);
	}

	if (mProperties.get() == nullptr)
	{
		mProperties = context->createBuffer(RT_BUFFER_INPUT);
		mProperties->setFormat(RT_FORMAT_USER);
		mProperties->setElementSize(sizeof(ScatteringMaterialProperties));
		mProperties->setSize(1);
	}



	reset();
}

bool BSSRDFCreator::on_draw(bool show_material_params)
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
	return changed;
}

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

HemisphereBSSRDFShader::HemisphereBSSRDFShader(const ShaderInfo & shader_info, std::unique_ptr<BSSRDFCreator>& creator, int camera_width, int camera_height) : Shader(shader_info),
mCameraWidth(camera_width),
mCameraHeight(camera_height)
{
	if (creator != nullptr)
		ref_impl = std::move(creator);
	else
		ref_impl = std::make_unique<ReferenceBSSRDF>(context);
}

void HemisphereBSSRDFShader::initialize_shader(optix::Context ctx)
{
	 Shader::initialize_shader(ctx);
	 //in static constructor

	 if (ref_impl == nullptr)
	 {
		 ref_impl = std::make_unique<PlanarBSSRDF>(context);

	 }

	 init_output();

	 reset();

}

void HemisphereBSSRDFShader::initialize_mesh(Mesh& object)
{

}

void HemisphereBSSRDFShader::pre_trace_mesh(Mesh& object)
{	
	
	ref_impl->render();

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

void PlanarBSSRDF::init()
{
	BSSRDFCreator::init();
	std::string ptx_path = get_path_ptx("planar_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["planar_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);
}

void PlanarBSSRDF::render()
{
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(20, 20, -1);
	context->launch(entry_point, mHemisphereSize.x, mHemisphereSize.y);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;
}

bool PlanarBSSRDF::on_draw(bool show_material_params)
{
	if (BSSRDFCreator::on_draw(show_material_params))
		reset();
	return false;
}

void PlanarBSSRDF::load_data()
{
	BSSRDFCreator::load_data();
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	cc->selected_bssrdf = mScatteringDipole;
	mProperties->unmap();
}

void ReferenceBSSRDF::init()
{
	BSSRDFCreator::init();
	std::string ptx_path = get_path_ptx("reference_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["reference_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);

}

void ReferenceBSSRDF::render()
{
	context->launch(entry_point, mSamples);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	mRenderedFrames++;
}

void ReferenceBSSRDF::load_data()
{
	BSSRDFCreator::load_data();
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);

}

bool ReferenceBSSRDF::on_draw(bool show_material_params)
{
	std::stringstream ss;
	ss << "Rendered: " << mRenderedFrames << " frames, " << mRenderedFrames*mSamples << " samples" << std::endl;
	ImmediateGUIDraw::Text(ss.str().c_str());
	bool changed = BSSRDFCreator::on_draw(show_material_params);
	changed |= ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples);
	changed |= ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations);
	if (changed)
		reset();
	return false;
}

void ReferenceBSSRDF::set_samples(int samples)
{
	mSamples = samples;
	reset();
}

void ReferenceBSSRDF::set_max_iterations(int max_iter)
{
	mMaxIterations = max_iter;
	reset();
}
