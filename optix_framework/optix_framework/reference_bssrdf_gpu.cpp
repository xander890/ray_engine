#include "reference_bssrdf_gpu.h"
#include "optix_utils.h"
#include "photon_trace_structs.h"

void ReferenceBSSRDFGPU::load_data()
{
}

void ReferenceBSSRDFGPU::initialize_shader(optix::Context ctx)
{
	Shader::initialize_shader(ctx);
	//in static constructor

	const std::string ptx_path = get_path_ptx("reference_bssrdf_gpu.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");

	entry_point = add_entry_point(context, ray_gen_program);

	init_output();

	context["resulting_flux"]->setBuffer(mBSSRDFBuffer);

	mAtomicPhotonCounterBuffer = create_and_initialize_buffer<int>(ctx, 0);

	mPhotonBuffer = create_buffer<PhotonSample>(ctx, mSamples);

}

void ReferenceBSSRDFGPU::initialize_mesh(Mesh& object)
{
}

void ReferenceBSSRDFGPU::pre_trace_mesh(Mesh& object)
{
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(0, -1, -1);
	context->launch(entry_point, mSamples);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;
}

void ReferenceBSSRDFGPU::post_trace_mesh(Mesh& object)
{
	ReferenceBSSRDF::post_trace_mesh(object);
}

bool ReferenceBSSRDFGPU::on_draw()
{
	return ReferenceBSSRDF::on_draw();
}

Shader* ReferenceBSSRDFGPU::clone()
{
	return new ReferenceBSSRDFGPU(*this);
}

ReferenceBSSRDFGPU::~ReferenceBSSRDFGPU()
{
}
