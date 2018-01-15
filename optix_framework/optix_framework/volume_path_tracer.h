#pragma once
#include "shader.h"
#include "immediate_gui.h"
#include "volume_path_tracing_common.h"

class VolumePathTracer : public Shader
{
public:
	virtual ~VolumePathTracer() = default;
	VolumePathTracer(const ShaderInfo& shader_info) : Shader(shader_info) {}

	virtual void VolumePathTracer::initialize_shader(optix::Context ctx) override
	{
		Shader::initialize_shader(ctx);
		max_vol_samples = ConfigParameters::get_parameter<int>("config", "maximum_volume_steps", max_vol_samples, "Maximum rays in VPT.");
	}

	virtual void load_data(Mesh & object) override
	{
		context["maximum_volume_steps"]->setUint(max_vol_samples);
		context["volume_pt_mode"]->setUint(mVolumePTMode);
	}

	virtual bool on_draw() override
	{
		bool changed = ImmediateGUIDraw::InputInt("Maximum rays in volume", (int*)&max_vol_samples);
		ImmediateGUIDraw::Text("Mode: ");
		ImmediateGUIDraw::SameLine();
		changed |= ImmediateGUIDraw::RadioButton("All", (int*)&mVolumePTMode, VOLUME_PT_INCLUDE_ALL);
		ImmediateGUIDraw::SameLine();
		changed |= ImmediateGUIDraw::RadioButton("Single scattering", (int*)&mVolumePTMode, VOLUME_PT_SINGLE_SCATTERING_ONLY);
		ImmediateGUIDraw::SameLine();
		changed |= ImmediateGUIDraw::RadioButton("Multiple scattering", (int*)&mVolumePTMode, VOLUME_PT_MULTIPLE_SCATTERING_ONLY);
		return changed;
	}

	virtual Shader* clone() override { return new VolumePathTracer(*this); }

	unsigned int max_vol_samples = 1000000;
	unsigned int mVolumePTMode = VOLUME_PT_INCLUDE_ALL;
};
