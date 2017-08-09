#pragma once
#include <vector>
#include "folders.h"
#include <optix_world.h>
#include <map>
#include <enums.h>

class ScatteringMaterial;
class Mesh;

struct MaterialData
{
    std::string name;
    optix::float3 emissive;
    optix::float3 reflectivity;
    optix::float3 absorption;
    float  phong_exp;
    float  ior;
    int    illum;
    optix::TextureSampler ambient_map;
    optix::TextureSampler diffuse_map;
    optix::TextureSampler specular_map;
    ScatteringMaterial * scattering_material;
};

class Shader
{
public:    
    friend class ShaderFactory;
    virtual ~Shader() = default;  
    virtual void initialize_mesh(Mesh & object) = 0;
    virtual void pre_trace_mesh(Mesh & object) = 0;  
    virtual void initialize_shader(optix::Context context, int illum);
    
protected:
    optix::Context context;
    int illum;
    RenderingMethodType::EnumType method;
    Shader() { }
    static void set_hit_programs(Mesh & object, std::string shader, RenderingMethodType::EnumType method);
};

