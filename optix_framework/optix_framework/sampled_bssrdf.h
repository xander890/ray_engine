#pragma once
#include "shader.h"
#include <memory>
#include <bssrdf_sampling_properties.h>
#include <optix_utils.h>

class SampledBSSRDF : public Shader
{
public:
	virtual ~SampledBSSRDF() = default;
	SampledBSSRDF(const ShaderInfo& shader_info);
	SampledBSSRDF(const SampledBSSRDF & cp);

	void initialize_shader(optix::Context ctx) override;
	void initialize_mesh(Mesh & object) override;
	void pre_trace_mesh(Mesh & object) override {}
	
	virtual bool on_draw() override;
	virtual void load_data() override;
	virtual Shader* clone() override { return new SampledBSSRDF(*this); }

	std::unique_ptr<BSSRDFSamplingProperties> properties = nullptr;
	optix::Buffer mPropertyBuffer;

	unsigned int mSamples = 25;
	bool mHasChanged = true;
};
