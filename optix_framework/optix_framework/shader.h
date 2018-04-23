#pragma once
#include <vector>
#include "folders.h"
#include "rendering_method.h"
#include <optix_world.h>
#include <map>
#include <enums.h>
#include <memory>

class Object;

struct ShaderInfo
{
	ShaderInfo(int illum, std::string path, std::string n) : illum(illum), cuda_shader_path(path), name(n) {}
    std::string cuda_shader_path;
    std::string name;
    int illum;
};

class Shader
{
public:    

	Shader(const ShaderInfo & info) : illum(info.illum), shader_path(info.cuda_shader_path),
		shader_name(info.name)
	{
	}

    friend class ShaderFactory;
    virtual ~Shader() = default;  

	virtual Shader* clone() = 0;
    virtual void initialize_mesh(Object &object);
    virtual void pre_trace_mesh(Object &object);
	virtual void post_trace_mesh(Object &object);
	virtual void load_data(Object &object) {}

    virtual void initialize_shader(optix::Context context);

	virtual bool on_draw();

    int get_illum() const { return illum; }
    std::string get_name() const { return shader_name; }
	void set_source(const std::string & source);

protected:
	Shader(const Shader & cp);

    void set_hit_programs(Object &object);

	optix::Context context;
	int illum;
	std::string shader_path;
	std::string shader_name;
};

