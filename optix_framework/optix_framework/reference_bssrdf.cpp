#include "reference_bssrdf.h"
#include "immediate_gui.h"
#include "optix_utils.h"

void ReferenceBSSRDF::initialize_shader(optix::Context ctx)
{
	 Shader::initialize_shader(ctx);
	 //in static constructor

	 std::string ptx_path = get_path_ptx("reference_bssrdf.cu");
	 optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");

	 entry_point = add_entry_point(context, ray_gen_program);

	 std::string ptx_path_output = get_path_ptx("render_reference_bssrdf.cu");
	 optix::Program ray_gen_program_output = context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	 entry_point_output = add_entry_point(context, ray_gen_program_output);

	 mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	 mBSSRDFBuffer->setFormat(RT_FORMAT_USER);
	 mBSSRDFBuffer->setElementSize(sizeof(float));
	 mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);

	 float* buff =reinterpret_cast<float*>(mBSSRDFBuffer->map());
	 memset(buff, 0, mHemisphereSize.x*mHemisphereSize.y * sizeof(float));
	 mBSSRDFBuffer->unmap();

	 context["resulting_flux"]->setBuffer(mBSSRDFBuffer);
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
}

void ReferenceBSSRDF::post_trace_mesh(Mesh & object)
{
	context->launch(entry_point_output, mHemisphereSize.x, mHemisphereSize.y);
}

bool ReferenceBSSRDF::on_draw()
{
	return false;
}