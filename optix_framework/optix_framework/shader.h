#pragma once
#include <vector>
#include "folders.h"
#include "rendering_method.h"
#include <optix_world.h>
#include <map>
#include <memory>

class MaterialHost;
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

	// Initializes mesh for rendering. Adds optix variables on the specific mesh/etc.
	// Executed once when the mesh is initialized with this particular shader.
    virtual void initialize_material(MaterialHost &object);

	// Executed every frame before tracing camera rays.
	// Loads appropriate data into the mesh for that specific frame.
	virtual void load_data(MaterialHost &object) {}

	// Executed every frame before tracing the camera rays.
	// Useful for some pre-processing on the mesh (sampling points, etc.)
    virtual void pre_trace_mesh(Object &object);

	// Executed every frame after tracing the camera rays.
	// Useful for post processing effects (gather data, etc.)
	virtual void post_trace_mesh(Object &object);

	// Removes variables when the mesh is destroyed.
	virtual void tear_down_material(MaterialHost &object);

    virtual void initialize_shader(optix::Context context);

	virtual bool on_draw();

    int get_illum() const { return info.illum; }
    std::string get_name() const { return info.shader_name; }
	void set_source(const std::string & source);

protected:
	Shader(const Shader & cp);
    Shader() {}

    void set_hit_programs(MaterialHost &mat);
	void remove_hit_programs(MaterialHost &mat);

	optix::Context context;
	ShaderInfo info;

private:

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        throw std::logic_error("Not implemented!");
    }

};

template<>
void Shader::serialize(cereal::XMLOutputArchiveOptix & archive);

template<>
void Shader::serialize(cereal::XMLInputArchiveOptix & archive);

