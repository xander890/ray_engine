#include "sampled_bssrdf.h"
#include "mesh.h"
#include "immediate_gui.h"

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
	}
	mHasChanged = false;
}

bool SampledBSSRDF::on_draw()
{
	std::vector<const char*> elems{ "Mertens et. al", "Hery et al." , "King et al." };
	
	mHasChanged |= ImmediateGUIDraw::Combo("Sampling technique", &properties->sampling_method, elems.data(), (int)elems.size(), (int)elems.size());
	if(properties->sampling_method == BSSRDF_SAMPLING_CAMERA_BASED_MERTENS)
		mHasChanged |= ImmediateGUIDraw::Checkbox("Jacobian", (bool*)&properties->correct_camera);
	if (properties->sampling_method != BSSRDF_SAMPLING_CAMERA_BASED_MERTENS)
		mHasChanged |= ImmediateGUIDraw::DragFloat("Radius max", &properties->R_max, 0.1f, 0.0f, 1.0f);
	
	mHasChanged |= ImGui::RadioButton("Show all", &properties->show_mode, BSSRDF_SHADERS_SHOW_ALL); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Refraction", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFRACTION); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Reflection", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFLECTION);
	return mHasChanged;
}

