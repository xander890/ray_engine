#pragma once
#include "shader.h"
#include "immediate_gui.h"
#include "volume_path_tracing_common.h"

class VolumePathTracer : public Shader
{
public:
	virtual ~VolumePathTracer() = default;
	VolumePathTracer(const ShaderInfo& shader_info) : Shader(shader_info) {}

	virtual void initialize_shader(optix::Context ctx) override
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
		changed |= ImmediateGUIDraw::CheckboxFlags("Direct transmission", &mVolumePTMode, VOLUME_PT_INCLUDE_DIRECT_TRANSMISSION);
		ImmediateGUIDraw::SameLine();
		changed |= ImmediateGUIDraw::CheckboxFlags("Single scattering", &mVolumePTMode, VOLUME_PT_INCLUDE_SINGLE_SCATTERING);
		ImmediateGUIDraw::SameLine();
		changed |= ImmediateGUIDraw::CheckboxFlags("Multiple scattering", &mVolumePTMode, VOLUME_PT_INCLUDE_MULTIPLE_SCATTERING);
		changed |= ImmediateGUIDraw::CheckboxFlags("Exclude backlit", &mVolumePTMode, VOLUME_PT_EXCLUDE_BACKLIT);

		return changed;
	}

	virtual Shader* clone() override { return new VolumePathTracer(*this); }

	unsigned int max_vol_samples = 1000000;
	unsigned int mVolumePTMode = VOLUME_PT_INCLUDE_DIRECT_TRANSMISSION | VOLUME_PT_INCLUDE_SINGLE_SCATTERING | VOLUME_PT_INCLUDE_MULTIPLE_SCATTERING;
};
