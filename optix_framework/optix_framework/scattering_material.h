#pragma once
#ifndef scattering_material_h__
#define scattering_material_h__
#include <memory>
#include <optix_world.h>
#include "structs.h"

#include "scattering_properties.h"

class MaterialHost;

enum DefaultScatteringMaterial
{
    Apple = 0,
    Marble = 1,
    Potato = 2,
    Skin = 3,
    ChocolateMilk = 4,
    Soymilk = 5,
    Whitegrapefruit = 6,
    ReducedMilk = 7,
    Ketchup = 8,
    Wholemilk = 9,
    Chicken = 10,
    Beer = 11,
    Coffee = 12,
    Shampoo = 13,
    Mustard = 14,
    MixedSoap = 15,
    GlycerineSoap = 16,
    Count = 17
};

class ScatteringMaterial
{
public:

    ScatteringMaterial(optix::float3 absorption,
                       optix::float3 scattering,
                       optix::float3 meancosine)
        : scale(1.0f), name("")
    {
        this->absorption = absorption;
        this->scattering = scattering;
        this->asymmetry = meancosine;
        mStandardMaterial = DefaultScatteringMaterial::Count; // Custom
		properties.selected_bssrdf = DIRECTIONAL_DIPOLE_BSSRDF;
		dirty = true;
    }

    ScatteringMaterial(DefaultScatteringMaterial material, float prop_scale = 100.0f)
    {
        scale = prop_scale;
        mStandardMaterial = static_cast<int>(material);
        getDefaultMaterial(material);
		properties.selected_bssrdf = DIRECTIONAL_DIPOLE_BSSRDF;
		dirty = true;
    }

    ScatteringMaterial& operator=(const ScatteringMaterial& cp);
    ScatteringMaterial(const ScatteringMaterial& cp);

    void getDefaultMaterial(DefaultScatteringMaterial material);
    void computeCoefficients(float relative_ior);

    float get_scale() const { return scale; }
    optix::float3 get_scattering() const { return scattering; }
    optix::float3 get_absorption() const { return absorption; }
    float get_asymmetry() const { return asymmetry.x; }

	void set_absorption(optix::float3 abs);
    void set_scattering(optix::float3 sc);
    void set_asymmetry(float asymm);
	bool on_draw(std::string id);

    const char* get_name() { return name; }
	bool hasChanged();
    ScatteringMaterialProperties get_data();

    static std::vector<ScatteringMaterial> defaultMaterials;

private:
    ScatteringMaterialProperties properties;
    float scale = 100.0f;
    optix::float3 scattering = optix::make_float3(0);
    optix::float3 asymmetry = optix::make_float3(0);
    optix::float3 absorption = optix::make_float3(1);

    const char* name;
    bool dirty = true;

    static std::vector<ScatteringMaterial> initializeDefaultMaterials();
    int mStandardMaterial;
};

#endif // scattering_material_h__
