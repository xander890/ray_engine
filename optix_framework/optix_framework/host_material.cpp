#include "host_material.h"
#include "parameter_parser.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>
#include "immediate_gui.h"
#include "obj_loader.h"

#include "optical_helper.h"

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
	bool changed = false;
	std::string myid = id + "Material" + to_string(mMaterialID);
	std::string newgroup = "Material " + to_string(mMaterialID) + "##" + myid;
	if (ImmediateGUIDraw::TreeNode(newgroup.c_str()))
	{
		changed |= scattering_material->on_draw(myid);
		if (ImmediateGUIDraw::InputFloat("Relative IOR", &mMaterialData.relative_ior))
		{
			changed = true;
			mHasChanged = true;
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

MaterialHost::MaterialHost(ObjMaterial& mat) : mMaterialName(), mMaterialData()
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
			Logger::error << "Need to define a user material if using illum -1" << endl;
		}
	}

	mMaterialName = data->name;
	std::transform(mMaterialName.begin(), mMaterialName.end(), mMaterialName.begin(), ::tolower);	
    static int id;
    mMaterialID = id++;

	mMaterialData.ambient_map = data->ambient_tex;
	mMaterialData.diffuse_map = data->diffuse_tex;
	mMaterialData.illum = data->illum;
	mMaterialData.shininess = data->shininess;
	mMaterialData.specular_map = data->specular_tex;

    Logger::info << "Looking for material properties for material " << mMaterialName << "..." << std::endl;
    ScatteringMaterial def = ScatteringMaterial(DefaultScatteringMaterial::Marble);
	
	if (findAndReturnMaterial(mMaterialName, def))
	{
		Logger::info << "Material found in default materials. " << std::endl;
		scattering_material = std::make_unique<ScatteringMaterial>(def);
		mMaterialData.relative_ior = data->ior == 0.0f ? 1.3f : data->ior;
		mMaterialData.ior_complex_imag_sq = optix::make_float3(mMaterialData.relative_ior*mMaterialData.relative_ior);
		mMaterialData.ior_complex_imag_sq = optix::make_float3(0);
		Logger::debug << to_string(scattering_material->get_scale()) << endl;
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

MaterialDataCommon& MaterialHost::get_data()
{
	if (mHasChanged || scattering_material->hasChanged())
	{
		scattering_material->computeCoefficients(mMaterialData.relative_ior);
		mHasChanged = false;
	}
    mMaterialData.scattering_properties = scattering_material->get_data();
    return mMaterialData;
}

MaterialDataCommon MaterialHost::get_data_copy()
{
	if (mHasChanged || scattering_material->hasChanged())
	{
		scattering_material->computeCoefficients(mMaterialData.relative_ior);
		mHasChanged = false;
	}
	mMaterialData.scattering_properties = scattering_material->get_data();
    return mMaterialData;
}

bool MaterialHost::hasChanged()
{
	return mHasChanged || scattering_material->hasChanged();
}

std::unique_ptr<ObjMaterial> MaterialHost::user_defined_material = nullptr;

void MaterialHost::set_default_material(ObjMaterial mat)
{
	user_defined_material = make_unique<ObjMaterial>(mat);
}
