#ifndef scattering_material_h__
#define scattering_material_h__

#include <optix_world.h>
#include "structs.h"
#include "GUI.h"
#include <functional>

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

    ScatteringMaterial(float indexOfRefraction,
                       optix::float3 absorption,
                       optix::float3 scattering,
                       optix::float3 meancosine)
        : scale(1.0f), name("")
    {
        this->ior = indexOfRefraction;
        this->absorption = absorption;
        this->scattering = scattering;
        this->asymmetry = meancosine;
        computeCoefficients();
    }

    ScatteringMaterial(DefaultScatteringMaterial material, float prop_scale = 100.0f)
    {
        scale = prop_scale;
        getDefaultMaterial(material);
    }

    ScatteringMaterial& operator=(const ScatteringMaterial& cp);

    void getDefaultMaterial(DefaultScatteringMaterial material);
    void loadParameters(const char* name, optix::GeometryInstance& context);
    void computeCoefficients();

    float get_scale() const { return scale; }
    optix::float3 get_scattering() const { return scattering; }
    optix::float3 get_absorption() const { return absorption; }
    float get_asymmetry() const { return asymmetry.x; }
    float get_ior() const { return ior; };

    void set_ior(float ior);
    void set_absorption(optix::float3 abs);
    void set_scattering(optix::float3 sc);
    void set_asymmetry(float asymm);
    void set_into_gui(GUI* gui);


private:
    ScatteringMaterialProperties properties;
    float scale = 1.0f;
    optix::float3 scattering = optix::make_float3(0);
    optix::float3 asymmetry = optix::make_float3(0);
    float ior = 1.0f;
    optix::float3 absorption = optix::make_float3(1);

    const char* name;
    bool dirty = true;

    static void GUI_CALL setDefault(const void* var, void* data);
    static void GUI_CALL getDefault(void* var, void* data);
    template <int channel>
    static void GUI_CALL setChannel(const void* var, void* data);
    template <int channel>
    static void GUI_CALL getChannel(void* var, void* data);
    static void GUI_CALL setAsymmetry(const void* var, void* data);
    static void GUI_CALL getAsymmetry(void* var, void* data);
    static void GUI_CALL setScale(const void* var, void* data);
    static void GUI_CALL getScale(void* var, void* data);
};

#endif // scattering_material_h__
