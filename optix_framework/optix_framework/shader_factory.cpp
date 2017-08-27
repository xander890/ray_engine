#include "shader_factory.h"
#include <optix_world.h>
#include "default_shader.h"
#include "material_library.h"
#include "glossy.h"
#include "../optprops/Medium.h"
#include "../optprops/spectrum2rgb.h"
#include "logger.h"
#include "mesh.h"
#include "presampled_surface_bssrdf.h"
#include "optix_helpers.h"
using namespace optix;

Context ShaderFactory::context = nullptr;



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


std::string get_full_program_name(std::string shader, std::string suffix)
{
    return shader + suffix;
}

const char * get_suffix(RenderingMethodType::EnumType method)
{
    switch (method)
    {
    case RenderingMethodType::RECURSIVE_RAY_TRACING: return "";
    case RenderingMethodType::AMBIENT_OCCLUSION: return "_ao";
    case RenderingMethodType::PATH_TRACING: return "_path_tracing";
    case RenderingMethodType::NotValidEnumItem: return "";
    default: return "";
    }

}

optix::Program ShaderFactory::createProgram(std::string file, std::string program_name, RenderingMethodType::EnumType m)
{
    const char * method = get_suffix(m);
    optix::Program result;
    try
    {
        Logger::warning << get_full_program_name(program_name, method) << std::endl;
        result = context->createProgramFromPTXFile(get_path_ptx(file), get_full_program_name(program_name, method));
    }
    catch (optix::Exception&)
    {
        Logger::warning << "Warning: function <" << get_full_program_name(program_name, method) << "> not found in file " << file << ". Reverting to full raytrace." << endl;
        // Fall back to standard ray tracing
        result = context->createProgramFromPTXFile(get_path_ptx(file), program_name);
    }
    return result;
}


void ShaderFactory::init(optix::Context& ctx)
{
    context = ctx;
    load_normalized_CIE_functions(context);

    ShaderInfo glossy = {"glossy_shader.cu", "Glossy", 2};
    add_shader<GlossyShader>(glossy);
    ShaderInfo bssrdf = { "subsurface_scattering_shader.cu", "Point cloud BSSRDF", 17 };
    add_shader<PresampledSurfaceBssrdf>(bssrdf);

    for (const ShaderInfo& n: DefaultShader::default_shaders)
    {
        DefaultShader* s = new DefaultShader();
        s->initialize_shader(context, n);
        mShaderMap[n.illum] = s;
    }
}

Shader* ShaderFactory::get_shader(int illum)
{
    std::string shader;
    if (mShaderMap.count(illum) != 0)
    {
        return mShaderMap[illum];
    }
    Logger::error << "Shader for illum " << illum << " not found" << std::endl;
    return nullptr;
}

map<int, Shader*> ShaderFactory::mShaderMap = map<int, Shader*>();