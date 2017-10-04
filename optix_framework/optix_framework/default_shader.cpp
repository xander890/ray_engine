#include "default_shader.h"

std::vector<ShaderInfo> DefaultShader::default_shaders = 
{
    {"constant_shader.cu" , "Constant" ,0},
    {"lambertian_shader.cu", "Lambertian" ,1},
    {"mirror_shader.cu", "Mirror" ,3},
    {"glass_shader.cu", "Glass" ,4},
	{ "normal_shader.cu", "Normals" ,5 },
	// {"dispersion_shader.cu", "Dispersion" ,6},
   // {"absorbing_glass.cu", "Absorption glass" ,7},
    {"metal_shader.cu", "Metal" ,11}
};

void DefaultShader::initialize_shader(optix::Context ctx)
{
    Shader::initialize_shader(ctx);
}

void DefaultShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
}

