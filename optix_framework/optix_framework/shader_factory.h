#pragma once
#include "shader.h"
#include <memory>
#include "logger.h"

class ShaderFactory
{
public:
    static void init(optix::Context& context);
    static std::unique_ptr<Shader> get_shader(int illum);
    static optix::Program createProgram(std::string file, std::string program_name, RenderingMethodType::EnumType method = RenderingMethodType::RECURSIVE_RAY_TRACING);
    template<typename T> static void add_shader(const ShaderInfo& shader_info);
	static const map<int, std::shared_ptr<Shader>>& get_map() { return mShaderMap; }

private:
    static optix::Context context;
    static map<int, std::shared_ptr<Shader>> mShaderMap;
    static map<int, std::shared_ptr<Shader>>& get_shader_map() { return mShaderMap; }
};

template <typename T>
void ShaderFactory::add_shader(const ShaderInfo& shader_info)
{
	if (get_shader_map().count(shader_info.illum) != 0)
	{
		Logger::warning << "Replacing shader! Be careful to know what you are doing!" << std::endl;
	}
    std::shared_ptr<Shader> s = make_shared<T>();
    s->initialize_shader(context, shader_info);
    get_shader_map()[shader_info.illum] = s;
}