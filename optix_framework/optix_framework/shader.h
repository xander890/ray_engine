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

	virtual Shader* clone() = 0;
    virtual void load_into_mesh(Mesh & object) = 0;
    virtual void pre_trace_mesh(Mesh & object) = 0;  

    virtual void initialize_shader(optix::Context context, const ShaderInfo& shader_info);
    void set_method(RenderingMethodType::EnumType m) { method = m; }

	virtual void set_into_gui(GUI * gui, const char * group = "");
	virtual void remove_from_gui(GUI * gui, const char * group = "");

	bool has_changed() { return mHasChanged; }

    int get_illum() const { return illum; }
    std::string get_name() const { return shader_name; }

protected:


    Shader(): illum(0), method()
    {
    }
	Shader(const Shader & cp);

    void set_hit_programs(Mesh & object);

	optix::Context context;
	int illum;
	std::string shader_path;
	std::string shader_name;
	RenderingMethodType::EnumType method = RenderingMethodType::PATH_TRACING;
	bool mHasChanged = true;
};

