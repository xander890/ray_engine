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

void get_relative_ior(const MPMLMedium& med_in, const MPMLMedium& med_out, optix::float3& eta, optix::float3& kappa)
{
    const float3& eta1 = med_in.ior_real;
    const float3& eta2 = med_out.ior_real;
    const float3& kappa1 = med_in.ior_imag;
    const float3& kappa2 = med_out.ior_imag;

    float3 ab = (eta1 * eta1 + kappa1 * kappa1);
    eta = (eta2 * eta1 + kappa2 * kappa1) / ab;
    kappa = (eta1 * kappa2 - eta2 * kappa1) / ab;
}

void Shader::initialize_mesh(Mesh & object)
{
    // Setting MPML data
    string n = object.get_main_material()->get_name();
    if (MaterialLibrary::media.count(n) != 0)
    {
        MPMLMedium mat = MaterialLibrary::media[n];
        object.mMaterial["mpml_material"]->setUserData(sizeof(MPMLMedium), &mat);
        MPMLMedium air = MaterialLibrary::media["air"];
        float3 eta, kappa;
        get_relative_ior(air, mat, eta, kappa);
        object.get_main_material()->get_data().ior_complex_real_sq = eta*eta;
        object.get_main_material()->get_data().ior_complex_imag_sq = kappa*kappa;

        Medium med = MaterialLibrary::full_media[n];
        //med.fill_spectral_data();
        Color<complex<double>> ior = med.get_ior(spectrum);
        if (ior.size() > 0)
        {
            Buffer b = object.mContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, ior.size());
            float * buff = (float *)b->map();
            for (int i = 0; i < ior.size(); i++) buff[i] = (float)ior[i].real();
            b->unmap();
            object.mMaterial["ior_real_spectrum"]->setBuffer(b);
            object.mMaterial["ior_real_wavelength"]->setFloat((float)ior.wavelength);
            object.mMaterial["ior_real_step"]->setFloat((float)ior.step_size);
        }
    }
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

void Shader::set_hit_programs(Mesh& object)
{
    auto chit = ShaderFactory::createProgram(shader_path, "shade", method);
    auto chitd = ShaderFactory::createProgram("depth_ray.cu", "depth");
    auto ahit = ShaderFactory::createProgram(shader_path, "any_hit_shadow");
    object.mMaterial->setClosestHitProgram(RAY_TYPE_RADIANCE, chit);
    object.mMaterial->setClosestHitProgram(RAY_TYPE_DEPTH, chitd);
    object.mMaterial->setAnyHitProgram(RAY_TYPE_SHADOW, ahit);
}


