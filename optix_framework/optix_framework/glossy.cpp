#include "glossy.h"
#include "folders.h"
#include "parameter_parser.h"
#include "mesh.h"

void GlossyShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
    float blinn_exponent = ParameterParser::get_parameter<float>("glossy", "blinn_exp", 1.0f);
    optix::float2 aniso_exp = ParameterParser::get_parameter<optix::float2>("glossy", "anisotropic_exp", optix::make_float2(.5f, 1.0f));
    optix::float3 xaxis = ParameterParser::get_parameter<optix::float3>("glossy", "x_axis_anisotropic", optix::make_float3(1.0f, 0.0f, 0.0f));

    object.mMaterial["exponent_blinn"]->setFloat(blinn_exponent);
    object.mMaterial["exponent_aniso"]->setFloat(aniso_exp);
    object.mMaterial["object_x_axis"]->setFloat(xaxis);

    set_hit_programs(object, "glossy_shader.cu", method);
}

void GlossyShader::pre_trace_mesh(Mesh& object)
{
    Shader::pre_trace_mesh(object);
}
