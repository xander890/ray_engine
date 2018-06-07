#include "host_material.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>
#include "immediate_gui.h"
#include "obj_loader.h"

#include "optical_helper.h"
#include "optix_utils.h"
#pragma warning (disable : 4244)
#pragma warning (disable : 4305)
#include <quantized_diffusion_helpers.h>
#include "logger.h"
#include "shader_factory.h"

using optix::float3;

bool findAndReturnMaterial(const std::string name, ScatteringMaterial & s)
{
	std::string to_compare = name;
	std::transform(to_compare.begin(), to_compare.end(), to_compare.begin(), ::tolower);
    auto ss = std::find_if(ScatteringMaterial::defaultMaterials.begin(), ScatteringMaterial::defaultMaterials.end(), [&](ScatteringMaterial & v){ return name.compare(v.get_name()) == 0; });
    if (ss != ScatteringMaterial::defaultMaterials.end())
        s = *ss;
    return ss != ScatteringMaterial::defaultMaterials.end();
}

bool MaterialHost::on_draw(std::string id = "")
{
	static bool first_time_gui = true;

	bool changed = false;
	std::string myid = id + "Material" + std::to_string(mMaterialID);
	std::string newgroup = mMaterialName + " (ID: " + std::to_string(mMaterialID) + ") ##" + myid;
	if (ImmediateGUIDraw::TreeNode(newgroup.c_str()))
	{

        auto map = ShaderFactory::get_map();
        std::vector<std::string> vv;
        std::vector<int> illummap;
        int initial = mShader->get_illum();
        int starting = 0;
        for (auto& p : map)
        {
            vv.push_back(p.second->get_name());
            illummap.push_back(p.first);
            if (p.first == initial)
                starting = (int)illummap.size()-1;
        }
        std::vector<const char *> v;
        for (auto& c : vv) v.push_back(c.c_str());

        int selected = starting;
        if (ImmediateGUIDraw::TreeNode((std::string("Shader##Shader") + mMaterialName).c_str()))
        {
            if (ImGui::Combo((std::string("Set Shader##RenderingMethod") + mMaterialName).c_str(), &selected, v.data(), (int)v.size(), (int)v.size()))
            {
                changed = true;
                set_shader(illummap[selected]);
            }

            bool shader_changed = mShader->on_draw();
			if(shader_changed)
				mReloadShader = true;


			changed |= shader_changed;
            ImmediateGUIDraw::TreePop();
        }

        if (ImmediateGUIDraw::TreeNode((std::string("Properties##Properties") + mMaterialName).c_str()))
        {
            static optix::float4 ka_gui = optix::make_float4(0, 0, 0, 1);
            static optix::float4 kd_gui = optix::make_float4(0, 0, 0, 1);
            static optix::float4 ks_gui = optix::make_float4(0, 0, 0, 1);
            if (first_time_gui)
            {
                get_texture_pixel<optix::float4>(mContext, ka_gui, mMaterialData.ambient_map);
                get_texture_pixel<optix::float4>(mContext, kd_gui, mMaterialData.diffuse_map);
                get_texture_pixel<optix::float4>(mContext, ks_gui, mMaterialData.specular_map);
                first_time_gui = false;
            }

            if (ImmediateGUIDraw::InputFloat3((std::string("Ambient##Ambient") + myid).c_str(), &ka_gui.x))
            {
                set_texture_pixel<optix::float4>(mContext, ka_gui, mMaterialData.ambient_map);
                if (optix::dot(optix::make_float3(ka_gui), optix::make_float3(1)) > 0)
                    mIsEmissive = true;
            }
            if (ImmediateGUIDraw::InputFloat3((std::string("Diffuse##Diffuse") + myid).c_str(), &kd_gui.x))
            {
                set_texture_pixel<optix::float4>(mContext, kd_gui, mMaterialData.diffuse_map);
            }
            if (ImmediateGUIDraw::InputFloat3((std::string("Specular##Specular") + myid).c_str(), &ks_gui.x))
            {
                set_texture_pixel<optix::float4>(mContext, ks_gui, mMaterialData.specular_map);
            }
            if (ImmediateGUIDraw::InputFloat((std::string("Relative IOR##IOR") +
                                              myid).c_str(), &mMaterialData.relative_ior))
            {
                changed = true;
            }
            changed |= scattering_material->on_draw(myid);
            ImmediateGUIDraw::TreePop();
        }
        ImmediateGUIDraw::TreePop();
	}
	return changed;
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

bool is_valid_material(ObjMaterial& mat)
{
	return mat.scale > 0.0f && dot(mat.absorption, optix::make_float3(1)) >= 0.0f && dot(mat.asymmetry, optix::make_float3(1)) >= 0.0f && dot(mat.scattering, optix::make_float3(1)) >= 0.0f;
}

MaterialHost::MaterialHost(optix::Context & context, ObjMaterial& mat) : MaterialHost(context)
{
	ObjMaterial * data = &mat;
	if (mat.illum == -1)
	{
		if (user_defined_material != nullptr)
		{
			data = user_defined_material.get();
		}
		else
		{
			Logger::error << "Need to define a user material if using illum -1" << std::endl;
		}
	}

	mMaterialName = data->name;
	std::transform(mMaterialName.begin(), mMaterialName.end(), mMaterialName.begin(), ::tolower);	


	mMaterialData.ambient_map = data->ambient_tex->get_id();
	mMaterialData.diffuse_map = data->diffuse_tex->get_id();
	mMaterialData.illum = data->illum;
	mMaterialData.shininess = data->shininess;
	mMaterialData.specular_map = data->specular_tex->get_id();

	textures.push_back(std::move(mat.ambient_tex));
	textures.push_back(std::move(mat.diffuse_tex));
	textures.push_back(std::move(mat.specular_tex));

	if (is_valid_material(*data))
	{
		Logger::info << mMaterialName << " is a valid obj material. Using obj parameters. " << std::endl;
		mMaterialData.relative_ior = mat.ior;
		scattering_material = std::make_unique<ScatteringMaterial>(mat.absorption, mat.scattering, mat.asymmetry, mat.scale, mMaterialName.c_str());
	}
	else
	{
		Logger::info << "Looking for material properties for material " << mMaterialName << "..." << std::endl;
		ScatteringMaterial def = ScatteringMaterial(DefaultScatteringMaterial::Marble);

		if (findAndReturnMaterial(mMaterialName, def))
		{
			Logger::info << "Material found in default materials. " << std::endl;
			scattering_material = std::make_unique<ScatteringMaterial>(def);
			mMaterialData.relative_ior = data->ior == 0.0f ? 1.3f : data->ior;
			mMaterialData.ior_complex_imag_sq = optix::make_float3(mMaterialData.relative_ior*mMaterialData.relative_ior);
			mMaterialData.ior_complex_imag_sq = optix::make_float3(0);
			Logger::debug << std::to_string(scattering_material->get_scale()) << std::endl;
		}
		else if (MaterialLibrary::media.count(mMaterialName) != 0)
		{
			Logger::info << "Material found in mpml file. " << std::endl;
			MPMLMedium mat = MaterialLibrary::media[mMaterialName];
			MPMLMedium air = MaterialLibrary::media["air"];
			float3 eta, kappa;
			get_relative_ior(air, mat, eta, kappa);
			mMaterialData.relative_ior = dot(eta, optix::make_float3(0.3333f));
			mMaterialData.ior_complex_real_sq = eta*eta;
			mMaterialData.ior_complex_imag_sq = kappa*kappa;
			scattering_material = std::make_unique<ScatteringMaterial>(mat.absorption, mat.scattering, mat.asymmetry);
		}
		else if (MaterialLibrary::interfaces.count(mMaterialName) != 0)
		{
			Logger::info << "Material found in mpml file as interface. " << std::endl;
			MPMLInterface interface = MaterialLibrary::interfaces[mMaterialName];
			float3 eta, kappa;
			get_relative_ior(*interface.med_out, *interface.med_in, eta, kappa);
			mMaterialData.relative_ior = dot(eta, optix::make_float3(0.3333f));
			mMaterialData.ior_complex_real_sq = eta*eta;
			mMaterialData.ior_complex_imag_sq = kappa*kappa;
			scattering_material = std::make_unique<ScatteringMaterial>(interface.med_in->absorption, interface.med_in->scattering, interface.med_in->asymmetry);
		}
		else
		{
			Logger::warning << "Scattering properties for material " << mMaterialName << "  not found. " << std::endl;
			mMaterialData.relative_ior = 1.0f;
			mMaterialData.ior_complex_imag_sq = optix::make_float3(mMaterialData.relative_ior*mMaterialData.relative_ior);
			scattering_material = std::make_unique<ScatteringMaterial>(optix::make_float3(1), optix::make_float3(0), optix::make_float3(1));
			mMaterialData.ior_complex_imag_sq = optix::make_float3(0);
		}
	}

	mIsEmissive = mat.is_emissive;

	if (!mMaterial)
	{
		mMaterial = mContext->createMaterial();
	}

    set_shader(mMaterialData.illum);

}


const MaterialDataCommon& MaterialHost::get_data()
{
	if (mHasChanged || scattering_material->hasChanged())
	{
		scattering_material->computeCoefficients(mMaterialData.relative_ior);
		mHasChanged = false;
		mMaterialData.scattering_properties = scattering_material->get_data();		
	}
    return mMaterialData;
}


bool MaterialHost::has_changed()
{
	return mHasChanged || scattering_material->hasChanged();
}

std::unique_ptr<ObjMaterial> MaterialHost::user_defined_material = nullptr;

void MaterialHost::set_default_material(ObjMaterial mat)
{
	user_defined_material = std::make_unique<ObjMaterial>(mat);
}

bool MaterialHost::is_emissive()
{
    return mIsEmissive;
}

MaterialHost::MaterialHost(optix::Context &ctx) : mContext(ctx), mMaterialName(), mMaterialData()
{
	static int id = 0;
	mMaterialID = id++;
	mMaterial = mContext->createMaterial();
}

void MaterialHost::reload_shader()
{
	mReloadShader = true;
}

void MaterialHost::set_shader(int illum)
{
	mShader = ShaderFactory::get_shader(illum);
	mReloadShader = true;
}


void MaterialHost::set_shader(const std::string &source) {
	if(mShader != nullptr)
	{
		mShader->set_source(source);
		mReloadShader = true;
	}
}

void MaterialHost::load_shader()
{
    if(mShader == nullptr)
    {
        set_shader(mMaterialData.illum);
    }

    if (mReloadShader)
    {
        mShader->initialize_material(*this);
        mReloadShader = false;
    }

    mShader->load_data(*this);
}

MaterialHost::~MaterialHost()
{
    if(mShader != nullptr && mMaterial.get() != nullptr)
    {
        mShader->tear_down_material(*this);
        mMaterial->destroy();
    }
}
