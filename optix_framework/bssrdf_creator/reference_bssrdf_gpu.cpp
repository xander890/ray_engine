#include "reference_bssrdf_gpu.h"
#include "optix_host_utils.h"
#include "photon_trace_structs.h"
#include "immediate_gui.h"
#include "string_utils.h"

void ReferenceBSSRDFGPU::init()
{
	if (mAtomicPhotonCounterBuffer.get() == nullptr)
	{		
		mAtomicPhotonCounterBuffer = create_buffer<int>(context, RT_BUFFER_INPUT_OUTPUT, mMaxFrames);
		mAtomicPhotonCounterBuffer->setFormat(RT_FORMAT_INT);
	}

	if (mPhotonBuffer.get() == nullptr)
	{
		mPhotonBuffer = create_buffer<PhotonSample>(context, RT_BUFFER_INPUT_OUTPUT, 10000000);
		mPhotonBuffer->setFormat(RT_FORMAT_USER);
		mPhotonBuffer->setElementSize(sizeof(PhotonSample));
	}

	BSSRDFRenderer::init();
	std::string ptx_path = Folders::get_path_to_ptx(ptx_file);

	if (entry_point == -1)
    {
        optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_gpu");
        entry_point = add_entry_point(context, ray_gen_program);
    }
	if (entry_point_post == -1)
    {
        optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_gpu_post");
        entry_point_post = add_entry_point(context, ray_gen_program_post);
    }

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

	float scattering = mAlbedo * mExtinction;
	set_bias(1.0f/(3.0f * scattering));

}

void ReferenceBSSRDFGPU::render()
{
	if (!mInitialized)
		init();
	if (mRenderedFrames >= mMaxFrames)
		return;

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

    BSSRDFRendererSimulated::render();

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
	BSSRDFRendererSimulated::load_data();
	context["batch_iterations"]->setUint(mBatchIterations);
}

void ReferenceBSSRDFGPU::set_samples(int samples)
{
	BSSRDFRendererSimulated::set_samples(samples);
}

bool ReferenceBSSRDFGPU::on_draw(unsigned int flags)
{
    bool changed = BSSRDFRendererSimulated::on_draw(flags);
	if (ImmediateGUIDraw::InputInt(GUI_LABEL("Batch iterations", mId), (int*)&mBatchIterations))
	{
        changed = true;
		reset();
	}
	ImmediateGUIDraw::InputInt(GUI_LABEL("Maximum frames: ", mId), (int*)&mMaxFrames);
	std::string photons = std::string("Photons cast: ") + std::to_string(mPhotons);
	ImmediateGUIDraw::Text("%s",photons.c_str());

	return changed;
}

size_t ReferenceBSSRDFGPU::get_samples()
{
	return mPhotons;
}

void ReferenceBSSRDFGPU::reset()
{
	BSSRDFRendererSimulated::reset();

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
