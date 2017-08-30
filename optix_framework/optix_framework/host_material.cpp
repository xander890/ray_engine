#include "host_material.h"
#include "parameter_parser.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>

bool findAndReturnMaterial(const std::string &name, ScatteringMaterial & s)
{
    auto ss = std::find_if(ScatteringMaterial::defaultMaterials.begin(), ScatteringMaterial::defaultMaterials.end(), [&](ScatteringMaterial & v){ return name.compare(v.get_name()) == 0; });
    if (ss != ScatteringMaterial::defaultMaterials.end())
        s = *ss;
    return ss != ScatteringMaterial::defaultMaterials.end();
}

void MaterialHost::set_into_gui(GUI* gui)
{
     scattering_material->set_into_gui(gui); 
}

void MaterialHost::remove_from_gui(GUI * gui)
{

}

MaterialHost::MaterialHost(const char * name, MaterialDataCommon data) : mMaterialName(name), mMaterialData(data)
{
    bool use_abs = ParameterParser::get_parameter("config", "use_absorption", true, "Use absorption in rendering.");
    if (!use_abs)
        mMaterialData.absorption = optix::make_float3(0.0f);


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
        Logger::error << "Material not found, defaulting to marble. " << std::endl;
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
