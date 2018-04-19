#pragma once
#include "shader.h"
#include <memory>
#include <bssrdf_properties.h>
#include <optix_utils.h>
#include <bssrdf_host.h>
#include "neural_network_sampler.h"

class SampledBSSRDF : public Shader
{

public:
	virtual ~SampledBSSRDF() = default;
	SampledBSSRDF(const ShaderInfo& shader_info);
	SampledBSSRDF(const SampledBSSRDF & cp);

	void initialize_shader(optix::Context ctx) override;
	void initialize_mesh(Object & object) override;
	void pre_trace_mesh(Object & object) override {}
	
	virtual bool on_draw() override;
	virtual void load_data(Object & object) override;
	virtual Shader* clone() override { return new SampledBSSRDF(*this); }

	std::unique_ptr<BSSRDFSamplingProperties> properties = nullptr;
	optix::Buffer mPropertyBuffer;

	unsigned int mSamples = 1;
	bool mHasChanged = true;
	bool mReloadShader = false;
    std::string mCurrentShaderSource;
	std::unique_ptr<BSSRDF> mBSSRDF;
    std::unique_ptr<NeuralNetworkSampler> mNNSampler;
	SamplingMfpType::Type mSamplingType = SamplingMfpType::MEAN;

private:
    std::string get_current_shader_source();
};
