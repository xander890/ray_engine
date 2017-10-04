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
	
	static void add_shader(std::unique_ptr<Shader> shader);
	static const std::map<int, std::shared_ptr<Shader>>& get_map() { return mShaderMap; }

private:
    static optix::Context context;
    static std::map<int, std::shared_ptr<Shader>> mShaderMap;
    static std::map<int, std::shared_ptr<Shader>>& get_shader_map() { return mShaderMap; }
};

