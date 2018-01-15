#include <shader.h>
#include "material_library.h"
#include "glossy.h"
#include "../optprops/Medium.h"
#include "host_device_common.h"
#include "mesh.h"
#include "default_shader.h"
#include "shader_factory.h"
#include "scattering_material.h"
#include "host_material.h"

using namespace optix;
using namespace std;


void Shader::initialize_mesh(Mesh & object)
{
    set_hit_programs(object);
}

void Shader::pre_trace_mesh(Mesh & object)
{
    return;
}

void Shader::post_trace_mesh(Mesh & object)
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
	method = cp.method;
}

void Shader::set_hit_programs(Mesh& object)
{
	Logger::info << "Loading closest hit programs..." << std::endl;
	auto chit = ShaderFactory::createProgram(shader_path, "shade", method);
    auto chitd = ShaderFactory::createProgram("depth_ray.cu", "depth");
	auto chita = ShaderFactory::createProgram("depth_ray.cu", "attribute_closest_hit");
	auto ahit = ShaderFactory::createProgram(shader_path, "any_hit_shadow");
    object.mMaterial->setClosestHitProgram( RayType::RADIANCE, chit);
    object.mMaterial->setClosestHitProgram( RayType::DEPTH, chitd);
	object.mMaterial->setClosestHitProgram( RayType::ATTRIBUTE, chita);
	object.mMaterial->setAnyHitProgram( RayType::SHADOW, ahit);
}

void Shader::set_source(const std::string &source) {
	shader_path = source;
}


