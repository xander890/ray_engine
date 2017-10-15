#include "reference_bssrdf_gpu.h"
#include "optix_utils.h"
#include "photon_trace_structs.h"
#include "immediate_gui.h"

void ReferenceBSSRDFGPU::init()
{
	if (mAtomicPhotonCounterBuffer.get() == nullptr)
	{
		mAtomicPhotonCounterBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, mMaxFrames);
	}

	if (mPhotonBuffer.get() == nullptr)
	{
		mPhotonBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, mSamples);
		mPhotonBuffer->setElementSize(sizeof(PhotonSample));
	}

	EmpiricalBSSRDFCreator::init();
	std::string ptx_path = get_path_ptx("reference_bssrdf_gpu.cu");
	
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_gpu");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_gpu_post");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);



	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["reference_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);
	BufPtr2D<float> ptr = BufPtr2D<float>(mBSSRDFBufferIntermediate->getId());
	context["reference_resulting_flux_intermediate"]->setUserData(sizeof(BufPtr2D<float>), &ptr);
	BufPtr2D<float> ptr2 = BufPtr2D<float>(mBSSRDFBuffer->getId());
	context["reference_resulting_flux"]->setUserData(sizeof(BufPtr2D<float>), &ptr2);

	BufPtr2D<PhotonSample> ptr3 = BufPtr2D<PhotonSample>(mPhotonBuffer->getId());
	context["photon_buffer"]->setUserData(sizeof(BufPtr2D<PhotonSample>), &ptr3);

	BufPtr2D<int> ptr4 = BufPtr2D<int>(mAtomicPhotonCounterBuffer->getId());
	context["photon_counter"]->setUserData(sizeof(BufPtr2D<int>), &ptr4);

}

void ReferenceBSSRDFGPU::render()
{
	if (!mInitialized)
		init();
	if (mRenderedFrames >= mMaxFrames)
		return;

	context->setPrintLaunchIndex(0,0,0);
	ReferenceBSSRDF::render();

	int * bufs = (int*)mAtomicPhotonCounterBuffer->map();
	mPhotons = 0;
	for (unsigned int i = 0; i < mRenderedFrames; i++)
	{
		mPhotons += bufs[i];
	}
	mAtomicPhotonCounterBuffer->unmap();
}

void ReferenceBSSRDFGPU::load_data()
{
	ReferenceBSSRDF::load_data();
	context["batch_iterations"]->setUint(mBatchIterations);
}

void ReferenceBSSRDFGPU::set_samples(int samples)
{
	mPhotonBuffer->setSize(samples);
	ReferenceBSSRDF::set_samples(samples);
}

bool ReferenceBSSRDFGPU::on_draw(bool show_material_params)
{
	ReferenceBSSRDF::on_draw(show_material_params);
	if (ImmediateGUIDraw::InputInt("Batch iterations", (int*)&mBatchIterations))
	{
		reset();
	}
	ImmediateGUIDraw::InputInt("Maximum frames: ", (int*)&mMaxFrames);
	std::string photons = std::string("Photons cast: ") + std::to_string(mPhotons) + " (Efficiency: " + std::to_string(((double)mPhotons)/(((double)mSamples)*(mRenderedFrames+1)));
	ImmediateGUIDraw::Text(photons.c_str());
	return false;
}

size_t ReferenceBSSRDFGPU::get_samples()
{
	return mPhotons;
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
	clear_buffer(mAtomicPhotonCounterBuffer);
}
