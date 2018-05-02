#pragma once
#ifndef scattering_material_h__
#define scattering_material_h__
#include <memory>
#include <optix_world.h>
#include "structs.h"
#include "scattering_properties.h"
#include "optix_serialize.h"
#include "enums.h"

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

    ScatteringMaterial(optix::float3 absorption = optix::make_float3(0.5f), optix::float3 scattering = optix::make_float3(0.5f), optix::float3 meancosine = optix::make_float3(0.0), float scale = 1.0f, const char * name = "");
	ScatteringMaterial(DefaultScatteringMaterial material, float prop_scale = 1.0f);

    ScatteringMaterial& operator=(const ScatteringMaterial& cp);
    ScatteringMaterial(const ScatteringMaterial& cp);

    void getDefaultMaterial(DefaultScatteringMaterial material);
    void computeCoefficients(float relative_ior);

    float get_scale() const { return scale; }
    optix::float3 get_scattering() const { return scattering; }
    optix::float3 get_absorption() const { return absorption; }
	optix::float3 get_asymmetry() const { return asymmetry; }

	void set_absorption(optix::float3 abs);
    void set_scattering(optix::float3 sc);
    void set_asymmetry(float asymm);
	bool on_draw(std::string id);

    const char* get_name() { return name; }
	bool hasChanged();
    ScatteringMaterialProperties get_data();

    static std::vector<ScatteringMaterial> defaultMaterials;
	static void initializeDefaultMaterials();

private:
    ScatteringMaterialProperties properties;
    float scale = 1.0f;
    optix::float3 scattering = optix::make_float3(0);
    optix::float3 asymmetry = optix::make_float3(0);
    optix::float3 absorption = optix::make_float3(1);

    const char* name;
    bool dirty = true;

    int mStandardMaterial;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(scale), CEREAL_NVP(scattering), CEREAL_NVP(asymmetry), CEREAL_NVP(absorption));
	}
};

#endif // scattering_material_h__
