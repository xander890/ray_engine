#pragma once
#include "shader.h"

class DefaultShader : public Shader
{
public:
    virtual ~DefaultShader() = default;
    DefaultShader() : Shader() {}
    
    void initialize_shader(optix::Context ctx, int illum) override;
    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override {}

    static std::map<int, std::string> default_shaders;
    std::string shader;
};
