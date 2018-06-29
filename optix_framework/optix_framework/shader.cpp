#include <shader.h>
#include "../optprops/Medium.h"
#include "object_host.h"
#include "shader_factory.h"
#include "scene.h"

void Shader::initialize_material(MaterialHost &object)
{
    set_hit_programs(object);
}

void Shader::pre_trace_mesh(Object &object)
{
    return;
}

void Shader::post_trace_mesh(Object &object)
{
	return;
}

void Shader::initialize_shader(optix::Context context)
{
    this->context = context;
}

bool Shader::on_draw()
{
	return false;
}

Shader::Shader(const Shader & cp)
{
	context = cp.context;
	info = cp.info;
}

void Shader::set_hit_programs(MaterialHost &mat)
{
	Logger::info << "Loading closest hit programs..." << std::endl;

	auto mRadianceClosestHit = ShaderFactory::createProgram(info.shader_path, "shade", mat.scene->get_method().get_suffix());
	auto mAnyHitProgram = ShaderFactory::createProgram(info.shader_path, "any_hit_shadow");

    auto mDepthClosestProgram = ShaderFactory::createProgram("util_rays.cu", "depth");
	auto mAttributeClosestProgram = ShaderFactory::createProgram("util_rays.cu", "attribute_closest_hit");
	auto mEmptyProgram = ShaderFactory::createProgram("util_rays.cu", "empty");

    mat.get_optix_material()->setClosestHitProgram(RayType::RADIANCE, mRadianceClosestHit);
	mat.get_optix_material()->setAnyHitProgram( RayType::RADIANCE, mEmptyProgram);

    mat.get_optix_material()->setClosestHitProgram(RayType::SHADOW, mEmptyProgram);
	mat.get_optix_material()->setAnyHitProgram(RayType::SHADOW, mAnyHitProgram);

	mat.get_optix_material()->setClosestHitProgram(RayType::DEPTH, mDepthClosestProgram);
	mat.get_optix_material()->setAnyHitProgram(RayType::DEPTH, mEmptyProgram);

	mat.get_optix_material()->setClosestHitProgram(RayType::ATTRIBUTE, mAttributeClosestProgram);
	mat.get_optix_material()->setAnyHitProgram(RayType::ATTRIBUTE, mEmptyProgram);

}

void Shader::set_source(const std::string &source) {
	info.shader_path = source;
}

Shader::~Shader()
{
}

void Shader::remove_hit_programs(MaterialHost &mat)
{
    if(mat.get_optix_material().get() == nullptr)
        return;

    auto program = mat.get_optix_material()->getClosestHitProgram(RayType::RADIANCE);
    if(program.get() != nullptr)
    {
        program->destroy();
    }

    program = mat.get_optix_material()->getAnyHitProgram( RayType::RADIANCE);
    if(program.get() != nullptr)
    {
        program->destroy();
    }

    program = mat.get_optix_material()->getAnyHitProgram(RayType::SHADOW);
    if(program.get() != nullptr)
    {
        program->destroy();
    }

    program = mat.get_optix_material()->getClosestHitProgram(RayType::DEPTH);
    if(program.get() != nullptr)
    {
        program->destroy();
    }

    program = mat.get_optix_material()->getClosestHitProgram(RayType::ATTRIBUTE);
    if(program.get() != nullptr)
    {
        program->destroy();
    }
}

void Shader::tear_down_material(MaterialHost &object)
{

    remove_hit_programs(object);
}

template<>
void Shader::serialize(cereal::XMLInputArchiveOptix & archive)
{
    context = archive.get_context();
    archive(cereal::make_nvp("name", info.shader_name), cereal::make_nvp("shader_path", info.shader_path), cereal::make_nvp("illum", info.illum));      
    
}

template<>
void Shader::serialize(cereal::XMLOutputArchiveOptix & archive)
{
    archive(cereal::make_nvp("type", std::string("expanded")));
    archive(cereal::make_nvp("name", info.shader_name), cereal::make_nvp("shader_path", info.shader_path), cereal::make_nvp("illum", info.illum));

}
