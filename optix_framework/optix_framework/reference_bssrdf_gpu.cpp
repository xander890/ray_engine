#include "reference_bssrdf_gpu.h"
#include "optix_utils.h"
#include "photon_trace_structs.h"
#include "immediate_gui.h"

void ReferenceBSSRDFGPU::load_data(Mesh & object)
{
}

void ReferenceBSSRDFGPU::initialize_shader(optix::Context ctx)
{
	Shader::initialize_shader(ctx);
	//in static constructor

	const std::string ptx_path = get_path_ptx("reference_bssrdf_gpu.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");

	entry_point = add_entry_point(context, ray_gen_program);

	init_output("render_reference_bssrdf_gpu.cu");

	mAtomicPhotonCounterBuffer = create_buffer<int>(ctx,1);
	mPhotonBuffer = create_buffer<PhotonSample>(ctx, mSamples);

	context["photon_counter"]->setBuffer(mAtomicPhotonCounterBuffer);
	context["photon_buffer"]->setBuffer(mPhotonBuffer);

	reset();
}

void ReferenceBSSRDFGPU::initialize_mesh(Mesh& object)
{
}

void ReferenceBSSRDFGPU::pre_trace_mesh(Mesh& object)
{
	context["batch_iterations"]->setUint(mBatchIterations);
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
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);
	if (ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples))
	{
		mPhotonBuffer->setSize(mSamples);
		reset();
	}
	
	if (ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations))
	{
		reset();
	}

	if (ImmediateGUIDraw::InputInt("Batch iterations", (int*)&mBatchIterations))
	{
		reset();
	}
	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);

	return false;
}

Shader* ReferenceBSSRDFGPU::clone()
{
	return new ReferenceBSSRDFGPU(*this);
}

ReferenceBSSRDFGPU::~ReferenceBSSRDFGPU()
{
}

void ReferenceBSSRDFGPU::reset()
{
	ReferenceBSSRDF::reset();
	PhotonSample * buf = reinterpret_cast<PhotonSample*>(mPhotonBuffer->map());
	PhotonSample start = get_empty_photon();
	for (unsigned int i = 0; i < mSamples; i++)
	{
		buf[i] = start;
	}
	mPhotonBuffer->unmap();
	initialize_buffer<int>(mAtomicPhotonCounterBuffer, 0);
}
