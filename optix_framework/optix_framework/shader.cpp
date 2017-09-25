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
    // Setting MPML data
    //string n = object.get_main_material()->get_name();
    //if (object.get_main_material()->get_data().illum == 5 && MaterialLibrary::media.count(n) != 0)
    //{
    //    Medium med = MaterialLibrary::full_media[n];
    //    med.fill_spectral_data();
    //    Color<complex<double>> ior = med.get_ior(spectrum);
    //    if (ior.size() > 0)
    //    {
    //        Buffer b = object.mContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, ior.size());
    //        float * buff = (float *)b->map();
    //        for (int i = 0; i < ior.size(); i++) buff[i] = (float)ior[i].real();
    //        b->unmap();
    //        object.mMaterial["ior_real_spectrum"]->setBuffer(b);
    //        object.mMaterial["ior_real_wavelength"]->setFloat((float)ior.wavelength);
    //        object.mMaterial["ior_real_step"]->setFloat((float)ior.step_size);
    //    }
    //}
    set_hit_programs(object);
}



void Shader::pre_trace_mesh(Mesh & object)
{
    return;
}

void Shader::initialize_shader(optix::Context context, const ShaderInfo& shader_info)
{
    this->context = context;
    illum = shader_info.illum;
    shader_path = shader_info.cuda_shader_path;
    shader_name = shader_info.name;
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
    auto chit = ShaderFactory::createProgram(shader_path, "shade", method);
    auto chitd = ShaderFactory::createProgram("depth_ray.cu", "depth");
	auto chita = ShaderFactory::createProgram("depth_ray.cu", "attribute_closest_hit");
	auto ahit = ShaderFactory::createProgram(shader_path, "any_hit_shadow");
    object.mMaterial->setClosestHitProgram( RayType::RADIANCE, chit);
    object.mMaterial->setClosestHitProgram( RayType::DEPTH, chitd);
	object.mMaterial->setClosestHitProgram( RayType::ATTRIBUTE, chita);
	object.mMaterial->setAnyHitProgram( RayType::SHADOW, ahit);
}


