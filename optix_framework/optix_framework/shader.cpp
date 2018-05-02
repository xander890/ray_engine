#include <shader.h>
#include "../optprops/Medium.h"
#include "object_host.h"
#include "shader_factory.h"
#include "scene.h"

void Shader::initialize_mesh(Object &object)
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
	illum = cp.illum;
	shader_path = cp.shader_path;
	shader_name = cp.shader_name;
}

void Shader::set_hit_programs(Object &object)
{
	Logger::info << "Loading closest hit programs..." << std::endl;

	auto mRadianceClosestHit = ShaderFactory::createProgram(shader_path, "shade", object.get_scene().get_method().get_suffix());
	auto mAnyHitProgram = ShaderFactory::createProgram(shader_path, "any_hit_shadow");

    auto mDepthClosestProgram = ShaderFactory::createProgram("util_rays.cu", "depth");
	auto mAttributeClosestProgram = ShaderFactory::createProgram("util_rays.cu", "attribute_closest_hit");
	auto mEmptyProgram = ShaderFactory::createProgram("util_rays.cu", "empty");

    object.mMaterial->setClosestHitProgram(RayType::RADIANCE, mRadianceClosestHit);
	object.mMaterial->setAnyHitProgram( RayType::RADIANCE, mEmptyProgram);

    object.mMaterial->setClosestHitProgram(RayType::SHADOW, mEmptyProgram);
	object.mMaterial->setAnyHitProgram(RayType::SHADOW, mAnyHitProgram);

	object.mMaterial->setClosestHitProgram(RayType::DEPTH, mDepthClosestProgram);
	object.mMaterial->setAnyHitProgram(RayType::DEPTH, mEmptyProgram);

	object.mMaterial->setClosestHitProgram(RayType::ATTRIBUTE, mAttributeClosestProgram);
	object.mMaterial->setAnyHitProgram(RayType::ATTRIBUTE, mEmptyProgram);

}

void Shader::set_source(const std::string &source) {
	shader_path = source;
}

Shader::~Shader()
{
}

void Shader::remove_hit_programs(Object &object)
{
    object.mMaterial->getClosestHitProgram(RayType::RADIANCE)->destroy();
    object.mMaterial->getAnyHitProgram( RayType::RADIANCE)->destroy();
    object.mMaterial->getAnyHitProgram(RayType::SHADOW)->destroy();
    object.mMaterial->getClosestHitProgram(RayType::DEPTH)->destroy();
    object.mMaterial->getClosestHitProgram(RayType::ATTRIBUTE)->destroy();
}


