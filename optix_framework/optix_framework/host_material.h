#pragma once
#include "material.h"
#include <memory>


class ScatteringMaterial;

class MaterialHost
{
public:
    MaterialHost(const char * name, MaterialDataCommon data);
    ~MaterialHost() = default;

	bool on_draw(std::string id);
    MaterialDataCommon& get_data(); 
    MaterialDataCommon get_data_copy();
    std::string get_name() { return mMaterialName; }
	bool hasChanged();

private:
    int mMaterialID;
    std::string mMaterialName;
    MaterialDataCommon mMaterialData;
    std::unique_ptr<ScatteringMaterial> scattering_material;
};

