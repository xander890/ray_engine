#pragma once
#include "shader.h"
#include "immediate_gui.h"

class VolumePathTracer : public Shader
{
public:
	virtual ~VolumePathTracer() = default;
	VolumePathTracer() : Shader() {}

	virtual void VolumePathTracer::initialize_shader(optix::Context ctx, const ShaderInfo& shader_info) override
	{
		Shader::initialize_shader(ctx, shader_info);
		max_vol_samples = ParameterParser::get_parameter<int>("config", "maximum_volume_steps", 1000000, "Maximum rays in VPT.");
	}

	virtual void load_data() override
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
