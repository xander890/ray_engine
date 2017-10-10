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
		max_vol_samples = ConfigParameters::get_parameter<int>("config", "maximum_volume_steps", 1000000, "Maximum rays in VPT.");
	}

	virtual void load_data(Mesh & object) override
	{
		context["maximum_volume_steps"]->setUint(max_vol_samples);
	}

	virtual bool on_draw() override
	{
		
		return ImmediateGUIDraw::InputInt("Maximum rays in volume", (int*)&max_vol_samples);
	}

	virtual Shader* clone() override { return new VolumePathTracer(*this); }

	unsigned int max_vol_samples;
};
