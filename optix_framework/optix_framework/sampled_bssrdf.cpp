#include "sampled_bssrdf.h"
#include "mesh.h"
#include "immediate_gui.h"
#include "scattering_material.h"

std::string sampling_method_to_string(int val)
{
	switch (val)
	{
	case  BSSRDF_SAMPLING_CAMERA_BASED: return "BSSRDF_SAMPLING_CAMERA_BASED";
	case  BSSRDF_SAMPLING_TANGENT_PLANE: return "BSSRDF_SAMPLING_TANGENT_PLANE";
	case  BSSRDF_SAMPLING_TANGENT_PLANE_TWO_PROBES: return "BSSRDF_SAMPLING_TANGENT_PLANE_TWO_PROBES";
	case  BSSRDF_SAMPLING_MIS_AXIS: return "BSSRDF_SAMPLING_MIS_AXIS";
	case  BSSRDF_SAMPLING_MIS_AXIS_AND_PROBES: return "BSSRDF_SAMPLING_MIS_AXIS_AND_PROBES";
	default:
	case  BSSRDF_SAMPLING_METHODS_COUNT: return "BSSRDF_SAMPLING_METHODS_COUNT";
	}
}

int sampling_method_to_enum(const std::string & val)
{
	for (int i = 0; i < BSSRDF_SAMPLING_METHODS_COUNT; i++)
		if (sampling_method_to_string(i) == val)
			return i;
	return BSSRDF_SAMPLING_METHODS_COUNT;
}


SampledBSSRDF::SampledBSSRDF() : Shader()
{
	properties = std::make_unique<BSSRDFSamplingProperties>();
}

SampledBSSRDF::SampledBSSRDF(const SampledBSSRDF & cp) : Shader(cp)
{
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
	properties = std::make_unique<BSSRDFSamplingProperties>();
	*properties = *cp.properties;
}

void SampledBSSRDF::initialize_shader(optix::Context ctx, const ShaderInfo& shader_info)
{
	Shader::initialize_shader(ctx, shader_info);
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
	properties->sampling_method = sampling_method_to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "sampling_method", sampling_method_to_string(properties->sampling_method), "Sampling method for illum 14."));
	Logger::info << "Using enum " << sampling_method_to_string(properties->sampling_method) << std::endl;
}

void SampledBSSRDF::initialize_mesh(Mesh& object)
{
	Shader::initialize_mesh(object);
	BufPtr<BSSRDFSamplingProperties> bufptr(mPropertyBuffer->getId());
 	object.mMaterial["bssrdf_sampling_properties"]->setUserData(sizeof(BufPtr<BSSRDFSamplingProperties>), &bufptr);

}

void SampledBSSRDF::load_data()
{
	if (mHasChanged)
	{
		Logger::info << "Reloading shader" << std::endl;
		initialize_buffer<BSSRDFSamplingProperties>(mPropertyBuffer, *properties);
		context["samples_per_pixel"]->setUint(mSamples);
	}
	mHasChanged = false;
}

bool SampledBSSRDF::on_draw()
{
	std::vector<const char*> elems{ "Camera based (Mertens et. al)", "Tangent plane (with distance)" , "Tangent plane (with probes)", "MIS axis (no probes)",  "MIS axis + probes (King et al.)" };
	
	assert(elems.size() == BSSRDF_SAMPLING_METHODS_COUNT);

	mHasChanged |= ImmediateGUIDraw::Combo("Sampling technique", &properties->sampling_method, elems.data(), (int)elems.size(), (int)elems.size());
	mHasChanged |= ImmediateGUIDraw::Checkbox("Jacobian", (bool*)&properties->use_jacobian);
	if (properties->sampling_method == BSSRDF_SAMPLING_TANGENT_PLANE)
	{
		mHasChanged |= ImmediateGUIDraw::DragFloat("Distance from surface", &properties->d_max, 0.1f, 0.0f, 1.0f);
		mHasChanged |= ImGui::InputFloat("Min no ni", &properties->dot_no_ni_min);
	}
	mHasChanged |= ImGui::RadioButton("Show all", &properties->show_mode, BSSRDF_SHADERS_SHOW_ALL); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Refraction", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFRACTION); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Reflection", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFLECTION);

	mHasChanged |= ImGui::InputInt("Samples per pixel", (int*)&mSamples);

	return mHasChanged;
}

