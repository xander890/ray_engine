#pragma once
#include "material_common.h"
#include <memory>
#include <optix_serialize_utils.h>
#include "texture.h"
#include "scattering_material.h"
#include "shader.h"
#include "shader_factory.h"

struct ObjMaterial;
class Scene;

namespace cereal
{
	template<class Archive>
	void serialize(Archive &archive, MaterialDataCommon &m)
	{ 
		archive(cereal::make_nvp("illum", m.illum));
		archive(cereal::make_nvp("index_of_refraction", m.index_of_refraction));
		archive(cereal::make_nvp("roughness", m.roughness));
		archive(cereal::make_nvp("anisotropy_angle", m.anisotropy_angle));
	}
}

class MaterialHost
{
public:
	MaterialHost(optix::Context& context, ObjMaterial& mat);
	MaterialHost(optix::Context& ctx);
	~MaterialHost();

	bool on_draw(std::string myid);
    const MaterialDataCommon& get_data(); 
    std::string get_name() { return mMaterialName; }
	bool has_changed();
	static void set_default_material(ObjMaterial mat);
	bool is_emissive();
	const std::vector<std::shared_ptr<Texture>>& get_textures() {return textures;}

	std::shared_ptr<Texture> get_ambient_texture() {return textures[0]; }
	std::shared_ptr<Texture> get_diffuse_texture() {return textures[1]; }
	std::shared_ptr<Texture> get_specular_texture() {return textures[2]; }

    void reload_shader();
    void set_shader(int illum);
    void set_shader(const std::string & source);
    void load_shader();

    Shader& get_shader() { return *mShader;}
    optix::Material& get_optix_material() { return mMaterial; }
    Scene * scene;

private:
	bool mHasChanged = true;
    int mMaterialID;
    std::string mMaterialName;
    MaterialDataCommon mMaterialData;
    std::unique_ptr<ScatteringMaterial> scattering_material;
	std::vector<std::shared_ptr<Texture>> textures;

    std::unique_ptr<Shader> mShader;
	bool mReloadShader = true;

	static std::unique_ptr<ObjMaterial> user_defined_material;

	friend class cereal::access;
	template<class Archive>
	void save(Archive & archive) const
	{
		archive(cereal::make_nvp("name", mMaterialName));
		archive(cereal::make_nvp("material_data", mMaterialData));
		archive(cereal::make_nvp("scattering_material",scattering_material));
		archive(cereal::make_nvp("is_emissive", mIsEmissive));
		archive(cereal::make_nvp("textures", textures));
        archive(cereal::make_nvp("shader_type", std::string("extended")));
        archive(cereal::make_nvp("shader", mShader));
    }

    static void load_and_construct(cereal::XMLInputArchiveOptix& archive, cereal::construct<MaterialHost>& construct);

	optix::Context mContext;
	bool mIsEmissive = false;
    optix::Material mMaterial = nullptr;

	bool first_time_gui = true;
	optix::float4 ka_gui = optix::make_float4(0, 0, 0, 1);
	optix::float4 kd_gui = optix::make_float4(0, 0, 0, 1);
	optix::float4 ks_gui = optix::make_float4(0, 0, 0, 1);

};


