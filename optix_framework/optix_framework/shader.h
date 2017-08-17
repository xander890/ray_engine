#pragma once
#include <vector>
#include "folders.h"
#include <optix_world.h>
#include <map>
#include <enums.h>
#include <scattering_material.h>

class Mesh;

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

