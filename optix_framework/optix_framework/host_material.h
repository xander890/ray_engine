#pragma once
#include "material.h"
#include <memory>
#include <optix_serialize.h>
#include "texture.h"
#include "scattering_material.h"
#include "shader.h"

struct ObjMaterial;

namespace cereal
{
	template<class Archive>
	void serialize(Archive &archive, MaterialDataCommon &m)
	{
		archive(cereal::make_nvp("illum", m.illum));
		archive(cereal::make_nvp("relative_ior", m.relative_ior));
		archive(cereal::make_nvp("shininess", m.shininess));
		archive(cereal::make_nvp("ior_complex_real_sq", m.ior_complex_real_sq));
		archive(cereal::make_nvp("ior_complex_imag_sq", m.ior_complex_imag_sq));
	}
}

class MaterialHost : public std::enable_shared_from_this<MaterialHost>
{
public:
	MaterialHost(optix::Context& ctx, ObjMaterial& data);

	bool on_draw(std::string id);
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
    void load_shader(Object &obj);

    Shader& get_shader() { return *mShader;}

private:
	MaterialHost(optix::Context& ctx);
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
        archive(cereal::make_nvp("shader", mShader));

    }

	static void load_and_construct( cereal::XMLInputArchiveOptix & archive, cereal::construct<MaterialHost> & construct )
	{
		construct(archive.get_context());
		archive(cereal::make_nvp("name", construct->mMaterialName));
		archive(cereal::make_nvp("material_data", construct->mMaterialData));
		archive(cereal::make_nvp("scattering_material", construct->scattering_material));
		archive(cereal::make_nvp("is_emissive", construct->mIsEmissive));
		archive(cereal::make_nvp("textures", construct->textures));
        archive(cereal::make_nvp("shader",construct->mShader));
        construct->mReloadShader = true;
        construct->mMaterialData.ambient_map = construct->textures[0]->get_id();
		construct->mMaterialData.diffuse_map = construct->textures[1]->get_id();
		construct->mMaterialData.specular_map = construct->textures[2]->get_id();
	}

	optix::Context mContext;
	bool mIsEmissive = false;
};


