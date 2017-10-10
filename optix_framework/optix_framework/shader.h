#pragma once
#include <vector>
#include "folders.h"
#include <optix_world.h>
#include <map>
#include <enums.h>

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

	Shader(const ShaderInfo & info) : illum(info.illum), shader_path(info.cuda_shader_path),
		shader_name(info.name), method()
	{
	}

    friend class ShaderFactory;
    virtual ~Shader() = default;  

	virtual Shader* clone() = 0;
    virtual void initialize_mesh(Mesh & object);
    virtual void pre_trace_mesh(Mesh & object);  
	virtual void post_trace_mesh(Mesh & object);
	virtual void load_data(Mesh & object) {}

    virtual void initialize_shader(optix::Context context);
    void set_method(RenderingMethodType::EnumType m) { method = m; }

	virtual bool on_draw();

    int get_illum() const { return illum; }
    std::string get_name() const { return shader_name; }

protected:


	Shader(const Shader & cp);

    void set_hit_programs(Mesh & object);

	optix::Context context;
	int illum;
	std::string shader_path;
	std::string shader_name;
	RenderingMethodType::EnumType method = RenderingMethodType::PATH_TRACING;

};

