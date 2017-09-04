#include "host_material.h"
#include "parameter_parser.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>

using optix::float3;

bool findAndReturnMaterial(const std::string &name, ScatteringMaterial & s)
{
    auto ss = std::find_if(ScatteringMaterial::defaultMaterials.begin(), ScatteringMaterial::defaultMaterials.end(), [&](ScatteringMaterial & v){ return name.compare(v.get_name()) == 0; });
    if (ss != ScatteringMaterial::defaultMaterials.end())
        s = *ss;
    return ss != ScatteringMaterial::defaultMaterials.end();
}

void MaterialHost::set_into_gui(GUI* gui, const char * group)
{
    std::string group_path = std::string(group);
    size_t last = group_path.find_last_of("/");
    std::string group_name = group_path.substr(last + 1);
    std::string newgroup = group_path + "/" + "Material (ID " + to_string(mMaterialID) + ")";    
    scattering_material->set_into_gui(gui, newgroup.c_str()); 
    gui->linkGroups(group, newgroup.c_str());
}

void MaterialHost::remove_from_gui(GUI * gui, const char * group)
{
	std::string group_path = std::string(group);
	size_t last = group_path.find_last_of("/");
	std::string group_name = group_path.substr(last + 1);
	std::string newgroup = group_path + "/" + "Material (ID " + to_string(mMaterialID) + ")";
	scattering_material->remove_from_gui(gui, newgroup.c_str());
}

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

MaterialHost::MaterialHost(const char * name, MaterialDataCommon data) : mMaterialName(name), mMaterialData(data)
{
    static int id;
    mMaterialID = id++;
    bool use_abs = ParameterParser::get_parameter("config", "use_absorption", true, "Use absorption in rendering.");
    if (!use_abs)
        mMaterialData.absorption = optix::make_float3(0.0f);

	if (MaterialLibrary::media.count(name) != 0)
	{
		MPMLMedium mat = MaterialLibrary::media[name];
		MPMLMedium air = MaterialLibrary::media["air"];
		float3 eta, kappa;
		get_relative_ior(air, mat, eta, kappa);
		mMaterialData.ior_complex_real_sq = eta*eta;
		mMaterialData.ior_complex_imag_sq = kappa*kappa;
	}
	else
	{
		mMaterialData.ior_complex_real_sq = optix::make_float3(1);
		mMaterialData.ior_complex_imag_sq = optix::make_float3(0);
	}

    Logger::info << "Looking for scattering material " << name << "..." << std::endl;
    ScatteringMaterial def = ScatteringMaterial(DefaultScatteringMaterial::Marble);
    if (MaterialLibrary::media.count(name) != 0)
    {
        Logger::info << "Material found in mpml file. " << std::endl;
        MPMLMedium mat = MaterialLibrary::media[name];
        scattering_material = std::make_unique<ScatteringMaterial>(mat.ior_real.x, mat.absorption, mat.scattering, mat.asymmetry);
    }
    else if (MaterialLibrary::interfaces.count(name) != 0)
    {
        Logger::info << "Material found in mpml file as interface. " << std::endl;
        MPMLInterface interface = MaterialLibrary::interfaces[name];
        float relative_index = interface.med_out->ior_real.x / interface.med_in->ior_real.x;
        scattering_material = std::make_unique<ScatteringMaterial>(relative_index, interface.med_in->absorption, interface.med_in->scattering, interface.med_in->asymmetry);
    }
    else if (findAndReturnMaterial(name, def))
    {
        Logger::info << "Material found in default materials. " << std::endl;
        scattering_material = std::make_unique<ScatteringMaterial>(def);
    }
    else
    {
        Logger::warning << "Scattering properties for material " << name << "  not found, defaulting to marble. " << std::endl;
        scattering_material = std::make_unique<ScatteringMaterial>(def);
    }

}

MaterialDataCommon& MaterialHost::get_data()
{
    mMaterialData.scattering_properties = scattering_material->get_data();
    return mMaterialData;
}

MaterialDataCommon MaterialHost::get_data_copy()
{
    mMaterialData.scattering_properties = scattering_material->get_data();
    return mMaterialData;
}

bool MaterialHost::hasChanged()
{
	return scattering_material->hasChanged();
}
