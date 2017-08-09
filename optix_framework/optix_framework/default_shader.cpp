#include "default_shader.h"

std::map<int, std::string> DefaultShader::default_shaders = 
{
    {0, "constant_shader.cu" },
    {1, "lambertian_shader.cu"},
    {3, "mirror_shader.cu"},
    {4, "glass_shader.cu"},
    {5, "dispersion_shader.cu"},
    {6, "absorbing_glass.cu"},
    {11, "metal_shader.cu"},
    {12, "volume_shader.cu"}
};

void DefaultShader::initialize_shader(optix::Context ctx, int illum)
{
    Shader::initialize_shader(ctx, illum);
    shader = default_shaders[illum];
}

void DefaultShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
    set_hit_programs(object, shader, method);
}

