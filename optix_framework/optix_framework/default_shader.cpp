#include "default_shader.h"

std::vector<ShaderInfo> DefaultShader::default_shaders = 
{
	ShaderInfo(0, "constant_shader.cu", "Constant"),
	ShaderInfo(1, "lambertian_shader.cu", "Lambertian"),
	ShaderInfo(3, "mirror_shader.cu", "Mirror"),
	ShaderInfo(4, "glass_shader.cu", "Glass"),
	ShaderInfo(5, "normal_shader.cu", "Normals"),
	//ShaderInfo(6, "dispersion_shader.cu", "Dispersion"),
	//ShaderInfo(7, "absorbing_glass.cu", "Absorption glass"),
	ShaderInfo(11, "metal_shader.cu", "Metal")
};

void DefaultShader::initialize_shader(optix::Context ctx)
{
    Shader::initialize_shader(ctx);
}

void DefaultShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
}

