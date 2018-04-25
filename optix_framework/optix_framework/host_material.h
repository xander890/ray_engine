#pragma once
#include "material.h"
#include <memory>
#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

struct ObjMaterial;
class ScatteringMaterial;
class Texture;

class MaterialHost : public std::enable_shared_from_this<MaterialHost>
{
public:
	MaterialHost(optix::Context& ctx, ObjMaterial& data);
    ~MaterialHost();

	bool on_draw(std::string id);
    const MaterialDataCommon& get_data(); 
    std::string get_name() { return mMaterialName; }
	bool hasChanged();
	static void set_default_material(ObjMaterial mat);

private:
	bool mHasChanged = true;
    int mMaterialID;
    std::string mMaterialName;
    MaterialDataCommon mMaterialData;
    std::unique_ptr<ScatteringMaterial> scattering_material;
	std::vector<std::shared_ptr<Texture>> textures;

	static std::unique_ptr<ObjMaterial> user_defined_material;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::make_nvp("name", mMaterialName), CEREAL_NVP(mMaterialData.illum), CEREAL_NVP(mMaterialData.relative_ior), CEREAL_NVP(mMaterialData.shininess), CEREAL_NVP(scattering_material));
	}
	optix::Context mContext;

	bool first_time_gui = true;
	optix::float4 ka_gui = optix::make_float4(0, 0, 0, 1);
	optix::float4 kd_gui = optix::make_float4(0, 0, 0, 1);
	optix::float4 ks_gui = optix::make_float4(0, 0, 0, 1);
};


