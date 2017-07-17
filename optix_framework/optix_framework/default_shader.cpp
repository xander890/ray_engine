#include "default_shader.h"


void DefaultShader::initialize_mesh(Mesh& object)
{
    Shader::initialize_mesh(object);
    std::string shader;
    if (!get_default_shader(illum, shader))
    {
        Logger::error << "Illum " << illum << " is not default. Exiting..." << std::endl;
        exit(2);
    }
    set_hit_programs(object, shader, method);
}

bool DefaultShader::get_default_shader(const int illum, std::string & shader)
{
    switch (illum)
    {
    case 0:
        shader = "constant_shader.cu";
        break;
    case 1:
        shader = "lambertian_shader.cu";
        break;
        //  see glossy.h
    case 3:
        shader = "mirror_shader.cu";
        break;
    case 4:
        shader = "glass_shader.cu";
        break;
    case 5:
        shader = "dispersion_shader.cu";
        break;
    case 6:
        shader = "absorbing_glass.cu";
        break;
    case 11:
        shader = "metal_shader.cu";
        break;
    default:
        return false;
    }
    return true;
}