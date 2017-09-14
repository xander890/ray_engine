#pragma once
#include "material.h"
#include <memory>
#include <cereal\access.hpp>
#include <cereal\cereal.hpp>

struct ObjMaterial;
class ScatteringMaterial;

class MaterialHost : std::enable_shared_from_this<MaterialHost>
{
public:
	MaterialHost() {}
	MaterialHost(ObjMaterial& data);
    ~MaterialHost();

	bool on_draw(std::string id);
    MaterialDataCommon& get_data(); 
    MaterialDataCommon get_data_copy();
    std::string get_name() { return mMaterialName; }
	bool hasChanged();
	static void set_default_material(ObjMaterial mat);

private:
	bool mHasChanged = true;
    int mMaterialID;
    std::string mMaterialName;
    MaterialDataCommon mMaterialData;
    std::unique_ptr<ScatteringMaterial> scattering_material;

	static std::unique_ptr<ObjMaterial> user_defined_material;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::make_nvp("name", mMaterialName), CEREAL_NVP(mMaterialData.illum), CEREAL_NVP(mMaterialData.relative_ior), CEREAL_NVP(mMaterialData.shininess), CEREAL_NVP(scattering_material));
	}
};


