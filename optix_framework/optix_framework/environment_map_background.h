#pragma once
#include "miss_program.h"
#include "environment_map.h"

class EnvironmentMap : public MissProgram
{
public:
    EnvironmentMap(std::string envmap_file) : envmap_file(envmap_file) {}
    virtual ~EnvironmentMap() {}

    virtual void init(optix::Context & ctx) override;
    virtual void set_into_gpu(optix::Context & ctx) override;
    virtual void set_into_gui(GUI * gui) override;
    virtual void remove_from_gui(GUI * gui) override;
private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;
    
    EnvmapProperties properties;
    EnvmapImportanceSamplingData sampling_properties;

    optix::float3 envmap_deltas;
    optix::TextureSampler environment_sampler;
    optix::Context context;
    optix::Buffer property_buffer;
    optix::Buffer sampling_property_buffer;
    std::string envmap_file;
    int texture_width, texture_height;

    static void GUI_CALL setDeltaX(const void* var, void* data);
    static void GUI_CALL getDeltaX(void* var, void* data);
    static void GUI_CALL setDeltaY(const void* var, void* data);
    static void GUI_CALL getDeltaY(void* var, void* data);
    static void GUI_CALL setDeltaZ(const void* var, void* data);
    static void GUI_CALL getDeltaZ(void* var, void* data);
    static void GUI_CALL setLightMultiplier(const void* var, void* data);
    static void GUI_CALL getLightMultiplier(void* var, void* data);

    void presample_environment_map();
    bool resample_envmaps = true;

};

