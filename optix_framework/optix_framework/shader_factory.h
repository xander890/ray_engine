#pragma once
#include "shader.h"

class ShaderFactory
{
public:
    static void init(optix::Context& context);
    static Shader* get_shader(int illum, RenderingMethodType::EnumType suffix);
    static optix::Program createProgram(std::string file, std::string program_name, RenderingMethodType::EnumType method = RenderingMethodType::RECURSIVE_RAY_TRACING);
    template<typename T> static void add_shader(int illum);
private:
    static optix::Context context;
    static map<int, Shader*> mShaderMap;
    static map<int, Shader*>& get_shader_map() { return mShaderMap; }
};

template <typename T>
void ShaderFactory::add_shader(int illum)
{
    Shader * s = new T();
    s->initialize_shader(context, illum);
    get_shader_map()[illum] = s;
}