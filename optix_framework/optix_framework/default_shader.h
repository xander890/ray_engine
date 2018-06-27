#pragma once
#include "shader.h"
#include "material_host.h"

/*
 * Common class for all the "standard" shaders, that do not require any particular gui or parameters.
 * The host behavior for these shaders it is the same, apart from the name, illum identifier and shader used. We discern these using the static mDefaultShaders table.
 */
class DefaultShader : public Shader
{
public:
    friend class ShaderFactory;
    virtual ~DefaultShader() = default;
    DefaultShader(const ShaderInfo& shader_info) : Shader(shader_info) {}
    
    void initialize_shader(optix::Context ctx) override;
    void initialize_material(MaterialHost &object) override;
    void pre_trace_mesh(Object & object) override {}

	Shader* clone() override { return new DefaultShader(*this); }

private:
    // This is only for cereal.
    DefaultShader() = default;

	friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        throw std::logic_error("Not implemented!");
    }

    // All the shaderinfo for common shaders. When implementing a new shader for the renderer, if can be initially added here. Otherwise see shader.h
    static std::vector<ShaderInfo> mDefaultShaders;
};

// Template explicit implementations for our custom archive.
template<>
void DefaultShader::serialize(cereal::XMLOutputArchiveOptix & archive);

template<>
void DefaultShader::serialize(cereal::XMLInputArchiveOptix & archive);

CEREAL_CLASS_VERSION(DefaultShader, 0)
CEREAL_REGISTER_TYPE(DefaultShader)