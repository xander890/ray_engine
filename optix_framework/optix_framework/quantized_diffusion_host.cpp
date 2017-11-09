#include "quantized_diffusion_host.h"
#include "quantized_diffusion_helpers.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "logger.h"

QuantizedDiffusion::QuantizedDiffusion(optix::Context & ctx) : BSSRDF(ctx, ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF)
{
	mBSSRDFPrecomputed = mContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mProperties.precomputed_bssrdf_size);
	auto id = mBSSRDFPrecomputed->getId();
	auto ptr_p = BufPtr1D<optix::float3>(id);
	mProperties.precomputed_bssrdf = ptr_p;

	mPropertyBuffer = create_and_initialize_buffer<QuantizedDiffusionProperties>(ctx, mProperties);
	auto ptr = BufPtr1D<QuantizedDiffusionProperties>(mPropertyBuffer->getId());
	mContext["qd_properties"]->setUserData(sizeof(BufPtr1D<QuantizedDiffusionProperties>), &ptr);
}

void QuantizedDiffusion::load(const ScatteringMaterialProperties & props)
{
	if (mHasChanged)
	{
		optix::float3 * buf = reinterpret_cast<optix::float3*>(mBSSRDFPrecomputed->map());
		Logger::info << "Precomputing quantized diffusion, please be patient..." << std::endl;
		precompute_quantized_diffusion(buf, mProperties.precomputed_bssrdf_size, mProperties.max_dist_bssrdf, props);
		mBSSRDFPrecomputed->unmap();
		mHasChanged = false;
		QuantizedDiffusionProperties * props = reinterpret_cast<QuantizedDiffusionProperties*>(mPropertyBuffer->map());
		*props = mProperties;
		mPropertyBuffer->unmap();

	}
}

void QuantizedDiffusion::on_draw()
{
	mHasChanged |= ImmediateGUIDraw::Checkbox("Use quantized", (bool*)&mProperties.use_precomputed_qd);
}
