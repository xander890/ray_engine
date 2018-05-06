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

	ShaderInfo(int illum = 0, std::string path = "", std::string n = "") : illum(illum), shader_path(path), shader_name(n) {}
    std::string shader_path;
    std::string shader_name;
    int illum;
};


class Shader
{
public:    
	Shader(const ShaderInfo & info) : info(info)
	{
	}

    friend class ShaderFactory;
    virtual ~Shader();

	virtual Shader* clone() = 0;
    virtual void initialize_mesh(Object &object);
    virtual void pre_trace_mesh(Object &object);
	virtual void post_trace_mesh(Object &object);
	virtual void load_data(Object &object) {}
	virtual void tear_down_mesh(Object &object) {remove_hit_programs(object);}

    virtual void initialize_shader(optix::Context context);

	virtual bool on_draw();

    int get_illum() const { return info.illum; }
    std::string get_name() const { return info.shader_name; }
	void set_source(const std::string & source);

protected:
	Shader(const Shader & cp);
    Shader() {}

    void set_hit_programs(Object &object);
	void remove_hit_programs(Object &object);

	optix::Context context;
	ShaderInfo info;

private:

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::make_nvp("name", info.shader_name),cereal::make_nvp("shader_path", info.shader_path), cereal::make_nvp("illum", info.illum));
    }

};

template<>
inline void Shader::serialize(cereal::XMLInputArchiveOptix & archive)
{
    archive(cereal::make_nvp("name", info.shader_name),cereal::make_nvp("shader_path", info.shader_path), cereal::make_nvp("illum", info.illum));
    context = archive.get_context();
}