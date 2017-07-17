#pragma once
#include "shader.h"

class DefaultShader : public Shader
{
public:
    virtual ~DefaultShader() = default;
    DefaultShader() : Shader() {}

    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override {}
    static bool default_shader_exists(int illum)
    {
        std::string s;
        return get_default_shader(illum, s);
    }

protected:
    static bool get_default_shader(const int illum, std::string & shader);
};
