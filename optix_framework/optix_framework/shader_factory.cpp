#include "shader_factory.h"
#include <optix_world.h>
#include "default_shader.h"
#include "material_library.h"
#include "brdf_shader.h"
#include "../optprops/Medium.h"
#include "../optprops/spectrum2rgb.h"
#include "logger.h"
#include "mesh.h"
#include "presampled_surface_bssrdf.h"
#include "optix_helpers.h"
#include "scattering_material.h"

optix::Context ShaderFactory::context = nullptr;

namespace optix
{
	void load_normalized_CIE_functions(optix::Context & ctx)
	{
		Color<double> spectrum_cdf(spectrum_rgb_samples);
		Color<float3> normalized_CIE_rgb(spectrum_rgb_samples);
		float3 cie_rgb = make_float3(0.0f);
		double cie = 0.0;
		for (unsigned int i = 0; i < spectrum_rgb_samples; ++i)
		{
			float3 rgb = make_float3(static_cast<float>(spectrum_rgb[i][1]), static_cast<float>(spectrum_rgb[i][2]), static_cast<float>(spectrum_rgb[i][3]));
			float rgb_sum = dot(rgb, make_float3(1.0f));
			cie += rgb_sum;
			normalized_CIE_rgb[i] = rgb;
			cie_rgb += rgb;
			spectrum_cdf[i] = cie;
		}
		double wavelength = spectrum_rgb[0][0];
		double step_size = spectrum_rgb[1][0] - spectrum_rgb[0][0];
		spectrum_cdf /= cie;
		spectrum_cdf.wavelength = wavelength;
		spectrum_cdf.step_size = step_size;
		normalized_CIE_rgb /= cie_rgb;
		normalized_CIE_rgb.wavelength = wavelength;
		normalized_CIE_rgb.step_size = step_size;

		Buffer ciergb = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, spectrum_rgb_samples);
		float3 * buff = (float3*)ciergb->map();
		for (unsigned int i = 0; i < spectrum_rgb_samples; ++i) buff[i] = normalized_CIE_rgb[i];
		ciergb->unmap();

		Buffer ciergbcdf = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, spectrum_rgb_samples);
		float * buff2 = (float*)ciergbcdf->map();
		for (unsigned int i = 0; i < spectrum_rgb_samples; ++i) buff2[i] = static_cast<float>(spectrum_cdf[i]);
		ciergbcdf->unmap();

		ctx["normalized_cie_rgb"]->setBuffer(ciergb);
		ctx["normalized_cie_rgb_cdf"]->setBuffer(ciergbcdf);
		ctx["normalized_cie_rgb_wavelength"]->setFloat(static_cast<float>(wavelength));
		ctx["normalized_cie_rgb_step"]->setFloat(static_cast<float>(step_size));
	}
}

std::string get_full_program_name(std::string shader, std::string suffix)
{
    return shader + suffix;
}

optix::Program ShaderFactory::createProgram(std::string file, std::string program_name, std::string m)
{
    const char * method = m.c_str();
    optix::Program result;
    try
    {
        std::string ptx_path = get_path_ptx(file);
        std::string full_name = get_full_program_name(program_name, method);
        result = context->createProgramFromPTXFile(ptx_path, full_name);
    }
    catch (optix::Exception&)
    {
        Logger::warning << "Warning: function <" << get_full_program_name(program_name, method) << "> not found in file " << file << ". Reverting to full raytrace." << std::endl;
        // Fall back to standard ray tracing
        result = context->createProgramFromPTXFile(get_path_ptx(file), program_name);
    }
    return result;
}

void ShaderFactory::add_shader(std::unique_ptr<Shader> shader)
{
	if (get_shader_map().count(shader->info.illum) != 0)
	{
		Logger::warning << "Replacing shader! Be careful to know what you are doing!" << std::endl;
	}
	std::shared_ptr<Shader> s = std::move(shader);
	get_shader_map()[s->info.illum] = s;
}

void ShaderFactory::init(optix::Context& ctx)
{
    context = ctx;
    load_normalized_CIE_functions(context);

    ShaderInfo glossy = ShaderInfo(2, "brdf_shader.cu", "BRDF");
    add_shader(std::make_unique<BRDFShader>(glossy));
    ShaderInfo bssrdf = ShaderInfo(17, "subsurface_scattering_shader.cu", "Point cloud BSSRDF"); 
	add_shader(std::make_unique<PresampledSurfaceBssrdf>(bssrdf));

    for (const ShaderInfo& n: DefaultShader::default_shaders)
    {
        add_shader(std::make_unique<DefaultShader>(n));
    }
}

std::unique_ptr<Shader> ShaderFactory::get_shader(int illum)
{
    std::string shader;
    if (mShaderMap.count(illum) != 0)
    {
		mShaderMap[illum]->initialize_shader(context);
        return std::unique_ptr<Shader>(mShaderMap[illum]->clone());
    }
    Logger::error << "Shader for illum " << illum << " not found" << std::endl;
    return nullptr;
}

std::map<int, std::shared_ptr<Shader>> ShaderFactory::mShaderMap = std::map<int, std::shared_ptr<Shader>>();