#pragma once
#include "shader.h"

class DefaultShader : public Shader
{
public:
    virtual ~DefaultShader() = default;
    DefaultShader(const ShaderInfo& shader_info) : Shader(shader_info) {}
    
    void initialize_shader(optix::Context ctx) override;
    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override {}

    static std::vector<ShaderInfo> default_shaders;
	virtual Shader* clone() override { return new DefaultShader(*this); }

};
