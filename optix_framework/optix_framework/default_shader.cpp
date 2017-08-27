#include "default_shader.h"

std::vector<ShaderInfo> DefaultShader::default_shaders = 
{
    {"constant_shader.cu" , "Constant" ,0},
    {"lambertian_shader.cu", "Lambertian" ,1},
    {"mirror_shader.cu", "Mirror" ,3},
    {"glass_shader.cu", "Glass" ,4},
    {"dispersion_shader.cu", "Dispersion" ,5},
    {"absorbing_glass.cu", "Absorption glass" ,6},
    {"metal_shader.cu", "Metal" ,11},
    {"volume_shader.cu", "Full volume PT" ,12}
};

void DefaultShader::initialize_shader(optix::Context ctx, const ShaderInfo& shader_info)
{
    Shader::initialize_shader(ctx, shader_info);
}

void DefaultShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
}

