#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"

void ReferenceBSSRDF::set_material_parameters(float albedo, float extinction, float g, float eta)
{
	mAlbedo = albedo;
	mExtinction = extinction;
	mAsymmetry = g;
	mIor = eta;
}

void ReferenceBSSRDF::init_output(const char * file)
{
	std::string ptx_path_output = get_path_ptx(file);
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	entry_point_output = add_entry_point(context, ray_gen_program_output);

	mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	mBSSRDFBuffer->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);

	mBSSRDFBufferTexture = context->createBuffer(RT_BUFFER_INPUT);
	mBSSRDFBufferTexture->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBufferTexture->setSize(mHemisphereSize.x, mHemisphereSize.y);

	mBSSRDFHemisphereTex = context->createTextureSampler();
	mBSSRDFHemisphereTex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	mBSSRDFHemisphereTex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	mBSSRDFHemisphereTex->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
	mBSSRDFHemisphereTex->setBuffer(mBSSRDFBufferTexture);
	mBSSRDFHemisphereTex->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
	mBSSRDFHemisphereTex->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
}

void ReferenceBSSRDF::reset()
{
	float* buff = reinterpret_cast<float*>(mBSSRDFBuffer->map());
	memset(buff, 0, mHemisphereSize.x*mHemisphereSize.y * sizeof(float));
	mBSSRDFBuffer->unmap();
	mRenderedFrames = 0;
}

void ReferenceBSSRDF::initialize_shader(optix::Context ctx)
{
	 Shader::initialize_shader(ctx);
	 //in static constructor

	 std::string ptx_path = get_path_ptx("reference_bssrdf.cu");
	 optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");

	 entry_point = add_entry_point(context, ray_gen_program);

	 init_output("render_reference_bssrdf.cu");

	 reset();

}

void ReferenceBSSRDF::initialize_mesh(Mesh& object)
{
}

void ReferenceBSSRDF::pre_trace_mesh(Mesh& object)
{	
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(0, -1, -1);
	context->launch(entry_point, mSamples);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;

	void* source = mBSSRDFBuffer->map();
	void* dest = mBSSRDFBufferTexture->map();
	memcpy(dest, source, mHemisphereSize.x*mHemisphereSize.y * sizeof(float));
	mBSSRDFBuffer->unmap();
	mBSSRDFBufferTexture->unmap();
}

void ReferenceBSSRDF::post_trace_mesh(Mesh & object)
{

	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool ReferenceBSSRDF::on_draw()
{
	if (ImmediateGUIDraw::SliderFloat("Incoming light angle (deg.)", &mThetai, 0, 90))
		reset();
	if (ImmediateGUIDraw::InputFloat("Radius", &mRadius))
		reset();
	if (ImmediateGUIDraw::SliderFloat("Angle on plane", &mThetas, 0, 360))
		reset();
	if (ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples))
		reset();
	if (ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations))
		reset();
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);

	ImmediateGUIDraw::Checkbox("Use parameters from material 0", &mUseMeshParameters);

	if (mUseMeshParameters)
	{
		std::stringstream ss;
		ss << "Currently used material properties:" << std::endl;
		ss << "Albedo: " << context["reference_bssrdf_albedo"]->getFloat() << std::endl;
		ss << "Extinction: " << context["reference_bssrdf_extinction"]->getFloat() << std::endl;
		ss << "Asymmetry: " << context["reference_bssrdf_g"]->getFloat() << std::endl;
		ss << "IOR: " << context["reference_bssrdf_rel_ior"]->getFloat() << std::endl;
		ImmediateGUIDraw::Text(ss.str().c_str());
	}
	else
	{
		ImmediateGUIDraw::InputFloat("Albedo##RefAlbedo", &mAlbedo);
		ImmediateGUIDraw::InputFloat("Extinction##RefExtinction", &mExtinction);
		ImmediateGUIDraw::InputFloat("G##RefAsymmetry", &mAsymmetry);
		ImmediateGUIDraw::InputFloat("Relative IOR##RefRelIOR", &mIor);
	}


	return false;
}

void ReferenceBSSRDF::load_data(Mesh & object)
{
	int s = mBSSRDFHemisphereTex->getId();
	context["resulting_flux"]->setBuffer(mBSSRDFBuffer);
	context["resulting_flux_tex"]->setUserData(sizeof(TexPtr),&(s));
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["show_false_colors"]->setUint(mShowFalseColors);
	context["ref_frame_number"]->setUint(mRenderedFrames);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);
	context["reference_scale_multiplier"]->setFloat(mScaleMultiplier);
	context["reference_bssrdf_theta_i"]->setFloat(mThetai);
	context["reference_bssrdf_theta_s"]->setFloat(mThetas);
	context["reference_bssrdf_radius"]->setFloat(mRadius);
	if(mUseMeshParameters)
	{
		context["reference_bssrdf_g"]->setFloat(object.get_main_material()->get_data().scattering_properties.meancosine.x);
		context["reference_bssrdf_albedo"]->setFloat(object.get_main_material()->get_data().scattering_properties.albedo.x);
		context["reference_bssrdf_extinction"]->setFloat(object.get_main_material()->get_data().scattering_properties.extinction.x);
		context["reference_bssrdf_rel_ior"]->setFloat(object.get_main_material()->get_data().relative_ior);
	}
	else
	{
		context["reference_bssrdf_g"]->setFloat(mAsymmetry);
		context["reference_bssrdf_albedo"]->setFloat(mAlbedo);
		context["reference_bssrdf_extinction"]->setFloat(mExtinction);
		context["reference_bssrdf_rel_ior"]->setFloat(mIor);
	}

}
