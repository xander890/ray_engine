#include "reference_bssrdf_gpu.h"
#include "optix_utils.h"
#include "photon_trace_structs.h"
#include "immediate_gui.h"

void ReferenceBSSRDFGPU::init()
{
	if (mAtomicPhotonCounterBuffer.get() == nullptr)
	{		
		mAtomicPhotonCounterBuffer = create_glbo_buffer<int>(context, RT_BUFFER_INPUT_OUTPUT, mMaxFrames);
		mAtomicPhotonCounterBuffer->setFormat(RT_FORMAT_INT);
	}

	if (mPhotonBuffer.get() == nullptr)
	{
		mPhotonBuffer = create_glbo_buffer<PhotonSample>(context, RT_BUFFER_INPUT_OUTPUT, 10000000);
		mPhotonBuffer->setFormat(RT_FORMAT_USER);
		mPhotonBuffer->setElementSize(sizeof(PhotonSample));
	}

	BSSRDFHemisphereRenderer::init();
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
	BSSRDFHemisphereSimulated::render();

	int * bufs = (int*)mAtomicPhotonCounterBuffer->map();
	mPhotons = 0;
	for (unsigned int i = 0; i < mRenderedFrames; i++)
	{
		mPhotons += (size_t)(bufs[i]);
	}
	mAtomicPhotonCounterBuffer->unmap();
}

void ReferenceBSSRDFGPU::load_data()
{
	BSSRDFHemisphereSimulated::load_data();
	context["batch_iterations"]->setUint(mBatchIterations);
}

void ReferenceBSSRDFGPU::set_samples(int samples)
{
	BSSRDFHemisphereSimulated::set_samples(samples);
}

bool ReferenceBSSRDFGPU::on_draw(bool show_material_params)
{
	BSSRDFHemisphereSimulated::on_draw(show_material_params);
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
	BSSRDFHemisphereSimulated::reset();

	PhotonSample * buf = reinterpret_cast<PhotonSample*>(mPhotonBuffer->map());
	PhotonSample start = get_empty_photon();
	RTsize s;
	mPhotonBuffer->getSize(s);
	for (unsigned int i = 0; i < s; i++)
	{
		buf[i] = start;
	}
	mPhotonBuffer->unmap();
	clear_buffer(mAtomicPhotonCounterBuffer);
}
