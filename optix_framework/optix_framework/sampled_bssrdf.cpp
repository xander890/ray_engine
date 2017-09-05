#include "sampled_bssrdf.h"
#include "mesh.h"

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

void SampledBSSRDF::load_into_mesh(Mesh& object)
{
	Shader::load_into_mesh(object);
	if (mHasChanged)
	{
		initialize_buffer<BSSRDFSamplingProperties>(mPropertyBuffer, *properties);
	}
	BufPtr<BSSRDFSamplingProperties> bufptr(mPropertyBuffer->getId());
	object.mMaterial["bssrdf_sampling_properties"]->setUserData(sizeof(BufPtr<BSSRDFSamplingProperties>), &bufptr);
	mHasChanged = false;
}

#define make_getter(type, var_name) \
[](void *var, void *clientData) \
{ \
	SampledBSSRDF * s = reinterpret_cast<SampledBSSRDF*>(clientData);\
	*((type*)var) = s->var_name;\
}

#define  make_setter(type, var_name, extra_code) \
[](const void *var, void *clientData) \
{ \
	SampledBSSRDF * s = reinterpret_cast<SampledBSSRDF*>(clientData);\
	s->var_name = *((type*)var); \
	extra_code \
}

void SampledBSSRDF::set_into_gui(GUI * gui, const char * group)
{
	std::string group_path = std::string(group);
	size_t last = group_path.find_last_of("/");
	std::string group_name = group_path.substr(last + 1);
	std::string newgroup = group_path + "/Sampling method";

	std::vector<GuiDropdownElement> elems{ {BSSRDF_SAMPLING_CAMERA_BASED_MERTENS, "Mertens et. al"}, {BSSRDF_SAMPLING_NORMAL_BASED_HERY, "Hery et al."}, {BSSRDF_SAMPLING_MIS_KING, "King et al."} };
	gui->addDropdownMenuCallback((newgroup + "/Method").c_str(), elems,
		make_setter(int, properties->sampling_method, s->mHasChanged = true;), make_getter(int, properties->sampling_method), this, newgroup.c_str());

	gui->addCheckBoxCallBack((newgroup + "/Use Jacobian").c_str(), 
		make_setter(int, properties->correct_camera, s->mHasChanged = true;), make_getter(int, properties->correct_camera), this, newgroup.c_str());

	gui->addFloatVariableCallBack((newgroup + "/R max").c_str(),
		make_setter(float, properties->R_max, s->mHasChanged = true;), make_getter(float, properties->R_max), this, newgroup.c_str());
	gui->linkGroups(group, newgroup.c_str());
}

void SampledBSSRDF::remove_from_gui(GUI * gui, const char * group)
{
	std::string group_path = std::string(group);
	std::string newgroup = group_path + "/Sampling method";
	gui->removeVar((newgroup + "/Method").c_str());
	gui->removeVar((newgroup + "/Use Jacobian").c_str());
	gui->removeVar((newgroup + "/R max").c_str());
}

