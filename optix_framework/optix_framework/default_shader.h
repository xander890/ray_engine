#pragma once
#include "shader.h"
#include "material_host.h"

class DefaultShader : public Shader
{
public:
    friend class ShaderFactory;
    virtual ~DefaultShader() = default;
    DefaultShader(const ShaderInfo& shader_info) : Shader(shader_info) {}
    
    void initialize_shader(optix::Context ctx) override;
    void initialize_material(MaterialHost &object) override;
    void pre_trace_mesh(Object & object) override {}

	virtual Shader* clone() override { return new DefaultShader(*this); }

private:
    DefaultShader() : Shader() {}
	friend class cereal::access;

    template<class Archive>
    void serialize(Archive & archive)
    {
        throw std::logic_error("Not implemented!");
    }

    static std::vector<ShaderInfo> default_shaders;
};

template<>
void DefaultShader::serialize(cereal::XMLOutputArchiveOptix & archive);

template<>
void DefaultShader::serialize(cereal::XMLInputArchiveOptix & archive);

CEREAL_CLASS_VERSION(DefaultShader, 0)
CEREAL_REGISTER_TYPE(DefaultShader)