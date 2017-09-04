#include "scattering_material.h"
#include "optical_helper.h"
#include <functional>

using namespace optix;

ScatteringMaterial& ScatteringMaterial::operator=(const ScatteringMaterial& cp)
{
    absorption = cp.absorption;
    ior = cp.ior;
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
    ior = cp.ior;
    scattering = cp.scattering;
    asymmetry = cp.asymmetry;
    name = cp.name;
    scale = cp.scale;
    mStandardMaterial = cp.mStandardMaterial;
	properties.selected_bssrdf = cp.properties.selected_bssrdf;
	dirty = true;
}

std::vector<ScatteringMaterial> ScatteringMaterial::initializeDefaultMaterials()
{
    std::vector<ScatteringMaterial> res;
    for (int i = 0; i < DefaultScatteringMaterial::Count; i++)
    {
        res.push_back(ScatteringMaterial(static_cast<DefaultScatteringMaterial>(i)));
    }
    return res;
}

std::vector<ScatteringMaterial> ScatteringMaterial::defaultMaterials = ScatteringMaterial::initializeDefaultMaterials();

void ScatteringMaterial::getDefaultMaterial(DefaultScatteringMaterial material)
{
  float3& sigma_a = absorption;
  float3& sigma_s = scattering;
  float3& g = asymmetry;
  sigma_a = make_float3(0.0f);
  sigma_s = make_float3(0.0f);
  g = make_float3(0.0f);
  ior = 1.3f;
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
    name = "whole milk";
    break;
  case Whitegrapefruit:
    sigma_a = make_float3(0.096f, 0.131f, 0.395f);
    sigma_s = make_float3(3.513f, 3.669f, 5.237f);
    g = make_float3(0.548f, 0.545f, 0.565f);    
    name = "white grapefruit";
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
    name = "soy milk";
  break;
  case Coffee:
    sigma_a = make_float3(0.1669f,0.2287f,0.3078f);
    sigma_s = make_float3(0.2707f,0.2828f,0.297f);
    g = make_float3(0.907f, 0.896f, 0.88f);
    name = "coffee";
  break;
  case Marble:
    ior = 1.5f;
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
    name = "chocolate milk";
  break;
    case ReducedMilk:
    sigma_a = make_float3(0.0001f, 0.0002f, 0.0005f);
    sigma_s = make_float3(10.748f, 12.209f, 13.931f);
    g = make_float3(0.819f, 0.797f, 0.746f);
    name = "reduced milk";
  break;
    case Mustard:
    sigma_s = make_float3(16.447f,18.536f,6.457f);
    sigma_a = make_float3(0.057f,0.061f,0.451f);
    g = make_float3(0.155f, 0.173f, 0.351f);
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
    name = "mixed soap";
    break;
    case GlycerineSoap:
    sigma_s = make_float3(0.201f, 0.202f, 0.221f);
    sigma_a = make_float3(0.001f, 0.001f, 0.002f);
    g = make_float3(0.955f, 0.949f, 0.943f);
    name = "glycerine soap";
    break;
  case Count: break;
  default: ;
  }
  dirty = true;
}

void ScatteringMaterial::set_ior(float ior)
{
	this->ior = ior;
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

void ScatteringMaterial::set_into_gui(GUI* gui, const char * group)
{
    std::vector<GuiDropdownElement> gui_elements;
    for (int i = 0; i < DefaultScatteringMaterial::Count; i++)
    {
        gui_elements.push_back({ i, defaultMaterials[i].name });
    }
    gui_elements.push_back({ DefaultScatteringMaterial::Count, "Custom" });

    std::string scamat = std::string(group);
    std::string group_path = std::string(group);
    size_t last = group_path.find_last_of("/");
    std::string gg = group_path.substr(last + 1);
    const char * group_name = group;

	std::vector<GuiDropdownElement> gui_elements_dipole = { {0, "Standard dipole"},{1 , "Directional dipole"} };

	gui->addDropdownMenuCallback((scamat + "/Set Dipole").c_str(), gui_elements_dipole, setBSSRDF, getBSSRDF, this, group_name);
	gui->addDropdownMenuCallback((scamat + "/Set Material").c_str(), gui_elements, setDefault, getDefault, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Absorption - R").c_str(), setAbsorptionChannel<0>, getAbsorptionChannel<0>, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Absorption - G").c_str(), setAbsorptionChannel<1>, getAbsorptionChannel<1>, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Absorption - B").c_str(), setAbsorptionChannel<2>, getAbsorptionChannel<2>, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Scattering - R").c_str(), setScatteringChannel<0>, getScatteringChannel<0>, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Scattering - G").c_str(), setScatteringChannel<1>, getScatteringChannel<1>, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Scattering - B").c_str(), setScatteringChannel<2>, getScatteringChannel<2>, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Asymmetry").c_str(), setAsymmetry, getAsymmetry, this, group_name);
    gui->addFloatVariableCallBack((scamat + "/Scale").c_str(), setScale, getScale, this, group_name);
}

void ScatteringMaterial::remove_from_gui(GUI* gui)
{
    gui->removeVar("Set Material");
    gui->removeVar("Absorption - R");
    gui->removeVar("Absorption - G");
    gui->removeVar("Absorption - B");
    gui->removeVar("Scattering - R");
    gui->removeVar("Scattering - G");
    gui->removeVar("Scattering - B");
    gui->removeVar("Asymmetry");
    gui->removeVar("Scale");
}


void GUI_CALL ScatteringMaterial::setBSSRDF(const void* var, void* data)
{
	ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
	s->properties.selected_bssrdf = (*(int*)var);
}

void GUI_CALL ScatteringMaterial::getBSSRDF(void* var, void* data)
{
	ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
	*(int*)var = s->properties.selected_bssrdf;
	s->dirty = true;
}

void ScatteringMaterial::setDefault(const void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    ScatteringMaterial newmat = ScatteringMaterial(static_cast<DefaultScatteringMaterial>(*(int*)var), s->scale);
    *s = newmat;
}

void ScatteringMaterial::getDefault(void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    *(int*)var = s->mStandardMaterial;
}

template<int channel> void ScatteringMaterial::setAbsorptionChannel(const void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    float * d = reinterpret_cast<float*>((float*)&s->absorption + channel);
    *d = *(float*)var;
    if (*d - *(float*)var < 1e-3)
        s->mStandardMaterial = Count;
}

template<int channel> void ScatteringMaterial::setScatteringChannel(const void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    float * d = reinterpret_cast<float*>((float*)&s->scattering + channel);
    *d = *(float*)var;
    if (*d - *(float*)var < 1e-3)
        s->mStandardMaterial = Count;
}


template <int channel>
void ScatteringMaterial::getScatteringChannel(void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    float * d = reinterpret_cast<float*>((float*)&s->scattering + channel);
    *(float*)var = *d;
}


template<int channel> void ScatteringMaterial::getAbsorptionChannel(void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    float * d = reinterpret_cast<float*>((float*)&s->absorption + channel);
    *(float*)var = *d;
}

void ScatteringMaterial::setAsymmetry(const void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    s->set_asymmetry(*(float*)var);
    s->dirty = true;
    if (s->asymmetry.x - *(float*)var < 1e-3)
        s->mStandardMaterial = Count;
}

void ScatteringMaterial::getAsymmetry(void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    *(float*)var = s->get_asymmetry();
}

void ScatteringMaterial::setScale(const void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    s->scale = *(float*)var;
    s->dirty = true;
}

void ScatteringMaterial::getScale(void* var, void* data)
{
    ScatteringMaterial * s = reinterpret_cast<ScatteringMaterial*>(data);
    *(float*)var = s->scale;
}

void ScatteringMaterial::computeCoefficients()
{
	properties.absorption = max(absorption, make_float3(1.0e-8f)) * scale;
	properties.scattering = scattering * scale;
	properties.meancosine = asymmetry;
	properties.relative_ior = ior;
	properties.deltaEddExtinction = properties.scattering*(1.0f - properties.meancosine*properties.meancosine) + properties.absorption;

	auto reducedScattering = properties.scattering * (make_float3(1.0f) - properties.meancosine);
	properties.reducedExtinction = reducedScattering + properties.absorption;
	properties.D = make_float3(1.0f) / (3.f * properties.reducedExtinction);
	properties.transport = sqrt(3*properties.absorption*properties.reducedExtinction);
	properties.C_phi = C_phi(properties.relative_ior);
	properties.C_phi_inv = C_phi(1.0f/properties.relative_ior);
	properties.C_E = C_E(properties.relative_ior);
	properties.reducedAlbedo = reducedScattering / properties.reducedExtinction;
	properties.de = 2.131f * properties.D / sqrt(properties.reducedAlbedo);
	properties.A = (1.0f - properties.C_E) / (2.0f * properties.C_phi);
	properties.extinction = properties.scattering + properties.absorption;
	properties.three_D = 3 * properties.D;
	properties.rev_D = (3.f * properties.reducedExtinction);
	properties.two_a_de = 2.0f * properties.A * properties.de;
	properties.global_coeff = 1.0f/(4.0f * properties.C_phi_inv) * 1.0f/(4.0f * M_PIf * M_PIf);
	properties.one_over_three_ext = make_float3(1.0) / (3.0f * properties.extinction);
	properties.albedo = properties.scattering / properties.extinction;
	//properties.two_de = 2.0f * properties.de;
	//properties.de_sqr = properties.de * properties.de;
	//properties.iorsq = properties.relative_ior * properties.relative_ior;
	properties.min_transport = fminf(fminf(properties.transport.x, properties.transport.y), properties.transport.z);
	properties.mean_transport = (properties.transport.x + properties.transport.y + properties.transport.z) / 3.0f;
	dirty = false;
}

bool ScatteringMaterial::hasChanged()
{
	return dirty;
}

ScatteringMaterialProperties ScatteringMaterial::get_data()
{
    if (dirty)
    {
        computeCoefficients();
    }
    return properties;
}