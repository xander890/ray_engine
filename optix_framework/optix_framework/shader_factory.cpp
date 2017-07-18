#include "shader_factory.h"
#include <optix_world.h>
#include "default_shader.h"
#include "material_library.h"
#include "glossy.h"
#include "../optprops/Medium.h"
#include "../optprops/spectrum2rgb.h"
#include "GEL/CGLA/Vec3f.h"
#include "logger.h"
#include "mesh.h"
#include "presampled_surface_bssrdf.h"

using namespace optix;

Context ShaderFactory::context = nullptr;

void load_normalized_CIE_functions(optix::Context & ctx)
{
    Color<double> spectrum_cdf(spectrum_rgb_samples);
    Color<CGLA::Vec3f> normalized_CIE_rgb(spectrum_rgb_samples);
    CGLA::Vec3f cie_rgb(0.0f);
    double cie = 0.0;
    for (unsigned int i = 0; i < spectrum_rgb_samples; ++i)
    {
        for (unsigned int j = 1; j < 4; ++j)
        {
            double rgb = spectrum_rgb[i][j];
            normalized_CIE_rgb[i][j - 1] = rgb;
            cie += rgb;
        }
        cie_rgb += normalized_CIE_rgb[i];
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
    for (unsigned int i = 0; i < spectrum_rgb_samples; ++i) buff[i] = make_float3(normalized_CIE_rgb[i][0], normalized_CIE_rgb[i][1], normalized_CIE_rgb[i][2]);
    ciergb->unmap();

    Buffer ciergbcdf = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, spectrum_rgb_samples);
    float * buff2 = (float*)ciergbcdf->map();
    for (unsigned int i = 0; i < spectrum_rgb_samples; ++i) buff2[i] = spectrum_cdf[i];
    ciergbcdf->unmap();

    ctx["normalized_cie_rgb"]->setBuffer(ciergb);
    ctx["normalized_cie_rgb_cdf"]->setBuffer(ciergbcdf);
    ctx["normalized_cie_rgb_wavelength"]->setFloat(wavelength);
    ctx["normalized_cie_rgb_step"]->setFloat(step_size);
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
    add_shader<GlossyShader>(2);
    add_shader<PresampledSurfaceBssrdf>(17);
}

Shader* ShaderFactory::get_shader(int illum, RenderingMethodType::EnumType method)
{
    std::string shader;
    if (mShaderMap.count(illum) != 0)
    {
        mShaderMap[illum]->method = method;
        return mShaderMap[illum];
    }
    if (DefaultShader::default_shader_exists(illum))
    {

        DefaultShader* s = new DefaultShader();
        s->initialize_shader(context, illum);
        s->method = method;       
        mShaderMap[illum] = s;
        return s;
    }
    Logger::error << "Shader for illum " << illum << " not found" << std::endl;
    return nullptr;
}

map<int, Shader*> ShaderFactory::mShaderMap = map<int, Shader*>();