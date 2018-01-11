#pragma once
#include "shader.h"
#include "immediate_gui.h"

class VolumePathTracer : public Shader
{
public:
	virtual ~VolumePathTracer() = default;
	VolumePathTracer(const ShaderInfo& shader_info) : Shader(shader_info) {}

	virtual void VolumePathTracer::initialize_shader(optix::Context ctx) override
	{
		Shader::initialize_shader(ctx);
		max_vol_samples = ConfigParameters::get_parameter<int>("config", "maximum_volume_steps", max_vol_samples, "Maximum rays in VPT.");
		mbSingleScatteringOnly = ConfigParameters::get_parameter<bool>("config", "maximum_volume_steps", mbSingleScatteringOnly, "Maximum rays in VPT.");
	}

	virtual void load_data(Mesh & object) override
	{
		if(mbSingleScatteringOnly)
		{
			context["maximum_volume_steps"]->setUint(2);
		}
		else
		{
			context["maximum_volume_steps"]->setUint(max_vol_samples);
		}
	}

	virtual bool on_draw() override
	{
		
		return ImmediateGUIDraw::InputInt("Maximum rays in volume", (int*)&max_vol_samples) | ImmediateGUIDraw::Checkbox("Single scattering only", &mbSingleScatteringOnly);
	}

	virtual Shader* clone() override { return new VolumePathTracer(*this); }

	unsigned int max_vol_samples = 1000000;
	bool mbSingleScatteringOnly = false;
};
