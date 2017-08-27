#pragma once
#include "shader.h"

class DefaultShader : public Shader
{
public:
    virtual ~DefaultShader() = default;
    DefaultShader() : Shader() {}
    
    void initialize_shader(optix::Context ctx, const ShaderInfo& shader_info) override;
    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override {}

    static std::vector<ShaderInfo> default_shaders;

};
