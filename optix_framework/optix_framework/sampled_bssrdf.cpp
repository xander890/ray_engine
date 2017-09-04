#include "sampled_bssrdf.h"
#include "mesh.h"

SampledBSSRDF::SampledBSSRDF() : Shader()
{
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
}

SampledBSSRDF::SampledBSSRDF(const SampledBSSRDF & cp)
{
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
	*properties = *cp.properties;
}

void SampledBSSRDF::initialize_shader(optix::Context ctx, const ShaderInfo& shader_info)
{
	Shader::initialize_shader(ctx, shader_info);
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
}

void SampledBSSRDF::set_into_gui(GUI * gui, const char * group)
{
}

void SampledBSSRDF::remove_from_gui(GUI * gui, const char * group)
{
}

