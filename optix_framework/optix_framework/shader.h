#pragma once
#include <vector>
#include "folders.h"
#include <optix_world.h>
#include <map>
#include <enums.h>
#include <scattering_material.h>

class Mesh;

struct ShaderInfo
{
    std::string cuda_shader_path;
    std::string name;
    int illum;
};

class Shader
{
public:    
    friend class ShaderFactory;
    virtual ~Shader() = default;  
    virtual void initialize_mesh(Mesh & object) = 0;
    virtual void pre_trace_mesh(Mesh & object) = 0;  
    virtual void initialize_shader(optix::Context context, const ShaderInfo& shader_info);
    void set_method(RenderingMethodType::EnumType m) { method = m; }
    int get_illum() const { return illum; }
    std::string get_name() const { return shader_name; }
	void reload(Mesh & object);

protected:
    optix::Context context;
    int illum;
    std::string shader_path;
    std::string shader_name;
    RenderingMethodType::EnumType method = RenderingMethodType::PATH_TRACING;

    Shader(): illum(0), method()
    {
    }

    void set_hit_programs(Mesh & object);
};

