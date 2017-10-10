#include "scattering_material.h"
#include "optical_helper.h"
#include <functional>
#include "immediate_gui.h"
#include "sampling_helpers.h"
#include "parameter_parser.h"
#include <algorithm>
#include "enums.h"

using namespace optix;


ScatteringMaterial::ScatteringMaterial(optix::float3 absorption, optix::float3 scattering, optix::float3 meancosine, float scale, const char * name)
	: scale(scale), name(name)
{
	this->absorption = absorption;
	this->scattering = scattering;
	this->asymmetry = meancosine;
	mStandardMaterial = DefaultScatteringMaterial::Count; // Custom
	properties.approx_property_A = ConfigParameters::get_parameter<optix::float3>("bssrdf", "approximate_A", make_float3(1), "Approximate value A for approximate dipoles.");
	properties.approx_property_s = ConfigParameters::get_parameter<optix::float3>("bssrdf", "approximate_s", make_float3(1), "Approximate value A for approximate dipoles.");
	properties.selected_bssrdf = ScatteringDipole::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_model", ScatteringDipole::to_string(properties.selected_bssrdf), (std::string("Selected dipole. Values: ") + ScatteringDipole::get_full_string()).c_str()));
	mSamplingType = SamplingMfpType::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_sampling_mfp", SamplingMfpType::to_string(mSamplingType), (std::string("Part of transport/s coeff. used for sampling. Values: ") + ScatteringDipole::get_full_string()).c_str()));
	dirty = true;
}

ScatteringMaterial::ScatteringMaterial(DefaultScatteringMaterial material, float prop_scale)
{
	scale = prop_scale;
	mStandardMaterial = static_cast<int>(material);
	getDefaultMaterial(material);
	properties.approx_property_A = ConfigParameters::get_parameter<optix::float3>("bssrdf", "approximate_A", properties.approx_property_A, "Approximate value A for approximate dipoles.");
	properties.approx_property_s = ConfigParameters::get_parameter<optix::float3>("bssrdf", "approximate_s", properties.approx_property_s, "Approximate value A for approximate dipoles.");
	properties.selected_bssrdf = ScatteringDipole::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_model", ScatteringDipole::to_string(properties.selected_bssrdf), (std::string("Selected dipole. Values: ") + ScatteringDipole::get_full_string()).c_str()));
	mSamplingType = SamplingMfpType::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_sampling_mfp", SamplingMfpType::to_string(mSamplingType), (std::string("Part of transport/s coeff. used for sampling. Values: ") + ScatteringDipole::get_full_string()).c_str()));
	dirty = true;
}

ScatteringMaterial& ScatteringMaterial::operator=(const ScatteringMaterial& cp)
{
    absorption = cp.absorption;
    scattering = cp.scattering;
    asymmetry = cp.asymmetry;
    name = cp.name;
    scale = cp.scale;
    dirty = true;
    mStandardMaterial = cp.mStandardMaterial;
	properties.selected_bssrdf = cp.properties.selected_bssrdf;
    return *this;
}

ScatteringMaterial::ScatteringMaterial(const ScatteringMaterial& cp)
{
    absorption = cp.absorption;
    scattering = cp.scattering;
    asymmetry = cp.asymmetry;
    name = cp.name;
    scale = cp.scale;
    mStandardMaterial = cp.mStandardMaterial;
	properties.selected_bssrdf = cp.properties.selected_bssrdf;
	dirty = true;
}

void ScatteringMaterial::initializeDefaultMaterials()
{
	defaultMaterials.clear();
    std::vector<ScatteringMaterial> res;
    for (int i = 0; i < DefaultScatteringMaterial::Count; i++)
    {
		defaultMaterials.push_back(ScatteringMaterial(static_cast<DefaultScatteringMaterial>(i)));
    }
}

std::vector<ScatteringMaterial> ScatteringMaterial::defaultMaterials;

void ScatteringMaterial::getDefaultMaterial(DefaultScatteringMaterial material)
{
  optix::float3& sigma_a = absorption;
  optix::float3& sigma_s = scattering;
  optix::float3& g = asymmetry;
  sigma_a = make_float3(0.0f);
  sigma_s = make_float3(0.0f);
  g = make_float3(0.0f);
  scale = 100.0f;

  switch(material)
  {
  case Chicken:
    sigma_a = make_float3(0.015f, 0.077f, 0.19f);
    sigma_s = make_float3(0.15f, 0.21f, 0.38f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "chicken";
  break;
  case Skin:
    sigma_a = make_float3(0.032f, 0.17f, 0.48f);
    sigma_s = make_float3(0.74f, 0.88f, 1.01f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "skin";
    break;
  case Wholemilk:
    sigma_a = make_float3(0.0011f,0.0024f,0.014f);
    sigma_s = make_float3(2.55f,3.21f,3.77f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "whole_milk";
    break;
  case Whitegrapefruit:
    sigma_a = make_float3(0.096f, 0.131f, 0.395f);
    sigma_s = make_float3(3.513f, 3.669f, 5.237f);
    g = make_float3(0.548f, 0.545f, 0.565f);   
    name = "white_grapefruit";
  break;
  case Beer:
    sigma_a = make_float3(0.1449f,0.3141f,0.7286f);
    sigma_s = make_float3(0.0037f,0.0069f,0.0074f);
    g = make_float3(0.917f, 0.956f, 0.982f);
    name = "beer";
  break;
  case Soymilk:
    sigma_a = make_float3(0.0001f,0.0005f,0.0034f);
    sigma_s = make_float3(2.433f,2.714f,4.563f);
    g = make_float3(0.873f, 0.858f, 0.832f);
    name = "soy_milk";
  break;
  case Coffee:
    sigma_a = make_float3(0.1669f,0.2287f,0.3078f);
    sigma_s = make_float3(0.2707f,0.2828f,0.297f);
    g = make_float3(0.907f, 0.896f, 0.88f);
    name = "coffee";
  break;
  case Marble:
    sigma_a = make_float3(0.0021f,0.0041f,0.0071f);
    sigma_s = make_float3(2.19f,2.62f,3.00f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "marble";
  break;
  case Potato:
    sigma_a = make_float3(0.0024f,0.0090f,0.12f);
    sigma_s = make_float3( 0.68f,0.70f,0.55f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "potato";
  break;
  case Ketchup:
    sigma_a = make_float3(0.061f,0.97f,1.45f);
    sigma_s = make_float3(0.18f,0.07f,0.03f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "ketchup";
  break;
  case Apple:
    sigma_a = make_float3(0.0030f,0.0034f,0.0046f);
    sigma_s = make_float3(2.29f,2.39f,1.97f);
    g = make_float3(0.0f, 0.0f, 0.0f);
    name = "apple";
  break;
    case ChocolateMilk:
    sigma_a = make_float3(0.007f, 0.03f, 0.1f);
    sigma_s = make_float3(7.352f, 9.142f, 10.588f);
    g = make_float3(0.862f, 0.838f, 0.806f);
	scale = 10.f;
    name = "chocolate_milk";
  break;
    case ReducedMilk:
    sigma_a = make_float3(0.0001f, 0.0002f, 0.0005f);
    sigma_s = make_float3(10.748f, 12.209f, 13.931f);
    g = make_float3(0.819f, 0.797f, 0.746f);
    name = "reduced_milk";
  break;
    case Mustard:
    sigma_s = make_float3(16.447f,18.536f,6.457f);
    sigma_a = make_float3(0.057f,0.061f,0.451f);
    g = make_float3(0.155f, 0.173f, 0.351f);
	scale = 1;
    name = "mustard";
    break;
    case Shampoo:
    sigma_s = make_float3(8.111f,9.919f,10.575f);
    sigma_a = make_float3(0.178f,0.328f,0.439f);
    g = make_float3(0.907f, 0.882f, 0.874f);
    name = "shampoo";
    break;
    case MixedSoap:
    sigma_s = make_float3(3.923f, 4.018f, 4.351f);
    sigma_a = make_float3(0.003f, 0.005f, 0.013f);
    g = make_float3(0.330f, 0.322f, 0.316f);
    name = "mixed_soap";
    break;
    case GlycerineSoap:
    sigma_s = make_float3(0.201f, 0.202f, 0.221f);
    sigma_a = make_float3(0.001f, 0.001f, 0.002f);
    g = make_float3(0.955f, 0.949f, 0.943f);
    name = "glycerine_soap";
    break;
  case Count: break;
  default: ;
  }
  dirty = true;
}

void ScatteringMaterial::set_absorption(optix::float3 abs)
{
	absorption = abs;
	dirty = true;
}

void ScatteringMaterial::set_scattering(optix::float3 sc)
{
	scattering = sc;
	dirty = true;
}

void ScatteringMaterial::set_asymmetry(float asymm)
{
	asymmetry = make_float3(asymm);
	dirty = true;
}


bool ScatteringMaterial::on_draw(std::string id)
{
	ImGui::Separator();
	ImGui::Text("Scattering properties:");
	const char * dips[ScatteringDipole::count()] = { "Standard dipole" , "Directional dipole", "Approximate Standard (Burley)", "Approximate Directional (Frisvad)" };
	#define ID_STRING(x,id) (std::string(x) + "##" + id + x).c_str()
	if (ImmediateGUIDraw::Combo(ID_STRING("Dipole", id), (int*)&properties.selected_bssrdf, dips, ScatteringDipole::count(), ScatteringDipole::count()))
	{
		dirty = true;
	}

	std::vector<std::string> vv;
	for (int i = 0; i < defaultMaterials.size(); i++)
	{
		vv.push_back(defaultMaterials[i].name);
	}
	vv.push_back("Custom");
	std::vector<const char*> v;
	for (auto& c : vv) v.push_back(c.c_str());
	static int mat = mStandardMaterial;

	bool is_approximate = properties.selected_bssrdf == ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF || properties.selected_bssrdf == ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF;
	if (is_approximate)
	{
		if (ImmediateGUIDraw::InputFloat3(ID_STRING("A", id), (float*)&properties.approx_property_A))
		{
			dirty = true;
			mat = DefaultScatteringMaterial::Count;
		}
		if (ImmediateGUIDraw::InputFloat3(ID_STRING("s", id), (float*)&properties.approx_property_s))
		{
			dirty = true;
			mat = DefaultScatteringMaterial::Count;
		}
	}
	else
	{

		if (ImmediateGUIDraw::InputFloat3(ID_STRING("Absorption", id), (float*)&absorption))
		{
			dirty = true;
			mat = DefaultScatteringMaterial::Count;
		}
		if (ImmediateGUIDraw::InputFloat3(ID_STRING("Scattering", id), (float*)&scattering))
		{
			dirty = true;
			mat = DefaultScatteringMaterial::Count;
		}

		if (ImmediateGUIDraw::InputFloat3(ID_STRING("Asymmetry", id), (float*)&asymmetry))
		{
			dirty = true;
			mat = DefaultScatteringMaterial::Count;
		}

		if (ImmediateGUIDraw::InputFloat(ID_STRING("Scale", id), (float*)&scale))
		{
			dirty = true;
		}
	}

	if (ImmediateGUIDraw::Combo(ID_STRING("Change material", id), &mat, v.data(), (int)v.size(), (int)v.size()))
	{
		if (mat < DefaultScatteringMaterial::Count)
		{
			dirty = true;
			mStandardMaterial = static_cast<int>(mat);
			getDefaultMaterial(static_cast<DefaultScatteringMaterial>(mat));
		}
	}


	return dirty;
#undef ID_STRING
}

float computeSamplingMfp(SamplingMfpType::Type e, const optix::float3& t)
{
	switch (e)
	{
	case SamplingMfpType::X: return t.x;
	case SamplingMfpType::Y: return t.y;
	case SamplingMfpType::Z: return t.z;
	case SamplingMfpType::MIN: return optix::fminf(t);
	case SamplingMfpType::MAX: return optix::fmaxf(t);
	case SamplingMfpType::MEAN: return optix::dot(t, optix::make_float3(0.333f));
	case SamplingMfpType::NotValidEnumItem: 
	default:
		return 0;
	}
}

void ScatteringMaterial::computeCoefficients(float ior)
{
	fill_scattering_parameters(properties, scale, ior, absorption, scattering, asymmetry);

	Logger::info << "Will sample using MFP: " << SamplingMfpType::to_string(mSamplingType) << std::endl;
	properties.sampling_mfp_tr = computeSamplingMfp(mSamplingType, properties.transport);
	properties.sampling_mfp_s = computeSamplingMfp(mSamplingType, properties.approx_property_s);
	dirty = false;
}

bool ScatteringMaterial::hasChanged()
{
	return dirty;
}

ScatteringMaterialProperties ScatteringMaterial::get_data()
{
    return properties;
}