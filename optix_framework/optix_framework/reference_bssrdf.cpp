#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"

void ReferenceBSSRDF::init_output(const char * file)
{
	std::string ptx_path_output = get_path_ptx(file);
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	entry_point_output = add_entry_point(context, ray_gen_program_output);

	mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	mBSSRDFBuffer->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);

	mBSSRDFBuffer2 = context->createBuffer(RT_BUFFER_INPUT);
	mBSSRDFBuffer2->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBuffer2->setSize(mHemisphereSize.x, mHemisphereSize.y);

	mBSSRDFBufferTex = context->createTextureSampler();
	mBSSRDFBufferTex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	mBSSRDFBufferTex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	mBSSRDFBufferTex->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
	mBSSRDFBufferTex->setBuffer(mBSSRDFBuffer2);
	mBSSRDFBufferTex->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
	mBSSRDFBufferTex->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
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
	void* dest = mBSSRDFBuffer2->map();
	memcpy(dest, source, mHemisphereSize.x*mHemisphereSize.y * sizeof(float));
	mBSSRDFBuffer->unmap();
	mBSSRDFBuffer2->unmap();
}

void ReferenceBSSRDF::post_trace_mesh(Mesh & object)
{

	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool ReferenceBSSRDF::on_draw()
{
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);
	if (ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples))
		reset();
	if (ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations))
		reset();
	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);
	return false;
}

void ReferenceBSSRDF::load_data()
{
	int s = mBSSRDFBufferTex->getId();
	context["resulting_flux"]->setBuffer(mBSSRDFBuffer);
	context["resulting_flux_tex"]->setUserData(sizeof(TexPtr),&(s));
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["show_false_colors"]->setUint(mShowFalseColors);
	context["ref_frame_number"]->setUint(mRenderedFrames);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);
	context["reference_scale_multiplier"]->setFloat(mScaleMultiplier);
}
