#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_utils.h"

void ReferenceBSSRDF::init_output(const char * file)
{
	std::string ptx_path_output = get_path_ptx(file);
	optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	entry_point_output = add_entry_point(context, ray_gen_program_output);

	mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	mBSSRDFBuffer->setFormat(RT_FORMAT_USER);
	mBSSRDFBuffer->setElementSize(sizeof(float));
	mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);
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

	 context["resulting_flux"]->setBuffer(mBSSRDFBuffer);

	 reset();

}

void ReferenceBSSRDF::initialize_mesh(Mesh& object)
{
}

void ReferenceBSSRDF::pre_trace_mesh(Mesh& object)
{
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["show_false_colors"]->setUint(mShowFalseColors);
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(0, -1, -1);
	context->launch(entry_point, mSamples);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;
}

void ReferenceBSSRDF::post_trace_mesh(Mesh & object)
{
	context["reference_bssrdf_samples_per_frame"]->setInt(mSamples * mRenderedFrames);
	context["reference_scale_multiplier"]->setFloat(mScaleMultiplier);

	context->launch(entry_point_output, mCameraWidth, mCameraHeight);
}

bool ReferenceBSSRDF::on_draw()
{
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);
	ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples);
	if (ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations))
		reset();
	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);
	return false;
}