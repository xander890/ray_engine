#pragma once
#include "shader.h"
#include "optix_serialize.h"
#include "brdf_host.h"
#include "host_material.h"

class BRDFShader : public Shader
{
public:
    virtual ~BRDFShader() = default;
    BRDFShader(const ShaderInfo& shader_info) : Shader(shader_info) {}
	BRDFShader(const BRDFShader &);

    void initialize_shader(optix::Context context) override;
    
    void initialize_material(MaterialHost &object) override;
    void pre_trace_mesh(Object & object) override;
	virtual Shader* clone() override { return new BRDFShader(*this); }
    bool on_draw() override;

private:

    BRDFShader() : Shader() {}

    // FIXME proper duplication.
    std::unique_ptr<BRDF> mBRDF;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::base_class<Shader>(this), mBRDF);
	}

};

CEREAL_REGISTER_TYPE(BRDFShader)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Shader, BRDFShader)
