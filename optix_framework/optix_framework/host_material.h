#pragma once
#include "material.h"
#include <memory>
#include "gui.h"

class ScatteringMaterial;

class MaterialHost
{
public:
    MaterialHost(const char * name, MaterialDataCommon data);
    ~MaterialHost() = default;

    void set_into_gui(GUI * gui);
    const MaterialDataCommon& get_data(); 
    MaterialDataCommon get_data_copy();
    std::string get_name() { return mMaterialName; }

private:
    std::string mMaterialName;
    MaterialDataCommon mMaterialData;
    std::unique_ptr<ScatteringMaterial> scattering_material;
};

