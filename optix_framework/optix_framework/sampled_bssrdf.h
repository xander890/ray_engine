#pragma once
#include "shader.h"
#include <memory>
#include <bssrdf_sampling_properties.h>
#include <optix_utils.h>

class SampledBSSRDF : public Shader
{
public:
	virtual ~SampledBSSRDF() = default;
	SampledBSSRDF();
	SampledBSSRDF(const SampledBSSRDF & cp);

	void initialize_shader(optix::Context ctx, const ShaderInfo& shader_info) override;
	void load_into_mesh(Mesh & object) override;
	void pre_trace_mesh(Mesh & object) override {}
	
	virtual void set_into_gui(GUI * gui, const char * group = "") override;
	virtual void remove_from_gui(GUI * gui, const char * group = "") override;
	virtual Shader* clone() override { return new SampledBSSRDF(*this); }

	std::unique_ptr<BSSRDFSamplingProperties> properties = nullptr;
	optix::Buffer mPropertyBuffer;
};
