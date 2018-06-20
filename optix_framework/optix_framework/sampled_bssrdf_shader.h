#pragma once
#include "shader.h"
#include <memory>
#include <bssrdf_common.h>
#include <optix_host_utils.h>
#include <bssrdf_host.h>
#include "neural_network_sampler.h"

#define IMPROVED_ENUM_NAME SamplingMfpType
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(X,0) ENUMITEM_VALUE(Y,1) ENUMITEM_VALUE(Z,2) ENUMITEM_VALUE(MEAN,3) ENUMITEM_VALUE(MIN,4) ENUMITEM_VALUE(MAX,5)
#include "improved_enum.inc"

class SampledBSSRDF : public Shader
{

public:
	virtual ~SampledBSSRDF() = default;
	SampledBSSRDF(const ShaderInfo& shader_info);
	SampledBSSRDF(const SampledBSSRDF & cp);

	void initialize_shader(optix::Context ctx) override;
	void initialize_material(MaterialHost &mat) override;
	void pre_trace_mesh(Object & object) override {}
	
	virtual bool on_draw() override;
	virtual void load_data(MaterialHost &mat) override;
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

    SampledBSSRDF() : Shader() {}
    friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::base_class<Shader>(this));
	}

};

CEREAL_CLASS_VERSION(SampledBSSRDF, 0)
CEREAL_REGISTER_TYPE(SampledBSSRDF)