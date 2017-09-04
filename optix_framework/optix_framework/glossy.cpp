#include "glossy.h"
#include "folders.h"
#include "parameter_parser.h"
#include "mesh.h"
#include "brdf_utils.h"
#include "merl_common.h"
#include <ImageLoader.h>
#include "host_material.h"
#include "scattering_material.h"

optix::TextureSampler createOneElementSampler(optix::Context context, const optix::float3& default_color)
{
    optix::TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setWrapMode(2, RT_WRAP_REPEAT);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    sampler->setMaxAnisotropy(1.0f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);

    // Create buffer with single texel set to default_color
    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1u, 1u);
    float* buffer_data = static_cast<float*>(buffer->map());
    buffer_data[0] = default_color.x;
    buffer_data[1] = default_color.y;
    buffer_data[2] = default_color.z;
    buffer_data[3] = 1.0f;
    buffer->unmap();

    sampler->setBuffer(0u, 0u, buffer);
    // Although it would be possible to use nearest filtering here, we chose linear
    // to be consistent with the textures that have been loaded from a file. This
    // allows OptiX to perform some optimizations.

    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

    return sampler;
}


void GlossyShader::initialize_shader(optix::Context context, const ShaderInfo& shader_info)
{
    Shader::initialize_shader(context, shader_info);
    blinn_exponent = ParameterParser::get_parameter<float>("glossy", "blinn_exp", 1.0f);
    anisotropic_exp = ParameterParser::get_parameter<optix::float2>("glossy", "anisotropic_exp", optix::make_float2(.5f, 1.0f));
    x_axis_anisotropic = ParameterParser::get_parameter<optix::float3>("glossy", "x_axis_anisotropic", optix::make_float3(1.0f, 0.0f, 0.0f));
    get_merl_brdf_list(Folders::merl_database_file.c_str(), brdf_names);
    merl_folder = Folders::merl_folder;
    merl_correction = ParameterParser::get_parameter<float3>("glossy", "merl_multiplier", make_float3(1.0f), "Multiplication factor for MERL materials. Premultiplied on sampling the brdf.");
    Logger::debug << "Merl correction factor: " << to_string(merl_correction.x) << " " << to_string(merl_correction.y) << " " << to_string(merl_correction.z) << endl;
    use_merl_brdf = ParameterParser::get_parameter<bool>("config", "use_merl_brdf", false, "configure the ray tracer to try to use the MERL brdf database whenever possible.");
}

void GlossyShader::set_data(Mesh& object)
{
    object.mMaterial["exponent_blinn"]->setFloat(blinn_exponent);
    object.mMaterial["exponent_aniso"]->setFloat(anisotropic_exp);
    object.mMaterial["object_x_axis"]->setFloat(x_axis_anisotropic);
    object.mMaterial["merl_brdf_multiplier"]->setFloat(merl_correction);
}

void GlossyShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
    set_data(object);
    string n = object.get_main_material()->get_name();

    string n_ext = n + ".binary";
    MERLBrdf * mat = nullptr;
    if (merl_database.count(n) != 0)
    {
        mat = &merl_database[n];
    }
    else if (std::find(brdf_names.begin(), brdf_names.end(), n_ext) != brdf_names.end())
    {
        merl_database[n] = MERLBrdf();
        mat = &merl_database[n];
        read_brdf_f(Folders::merl_folder, n_ext, mat->data);
        mat->name = n;
        mat->reflectance = integrate_brdf(mat->data, 100000);
    }
    else
    {
        Logger::warning << "Equivalent MERL Material " << n << " not found." << std::endl;
    }
    auto optix_mat = object.mMaterial;

    uint has_merl = mat == nullptr ? 0 : 1;
    float3 reflectance = mat == nullptr ? make_float3(0) : mat->reflectance;
    size_t buffer_size = mat == nullptr ? 1 : mat->data.size();


    optix_mat["has_merl_brdf"]->setUint(has_merl);
    optix::Buffer buff = optix_mat->getContext()->createBuffer(RT_BUFFER_INPUT);
    buff->setFormat(RT_FORMAT_FLOAT);
    buff->setSize(buffer_size);
    if (mat != nullptr)
    {
        void* b = buff->map();
        memcpy(b, mat->data.data(), buffer_size * sizeof(float));
        buff->unmap();
    }
    optix_mat["merl_brdf_buffer"]->setBuffer(buff);
    optix_mat["diffuse_map"]->setTextureSampler(createOneElementSampler(optix_mat->getContext(), reflectance));
    optix_mat["merl_brdf_buffer"]->setBuffer(buff);

}

void GlossyShader::pre_trace_mesh(Mesh& object)
{
    Shader::pre_trace_mesh(object);
    set_data(object);
}
