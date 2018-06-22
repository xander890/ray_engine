#pragma once
#include "shader.h"
#include "optix_serialize_utils.h"
#include "brdf_host.h"
#include "material_host.h"

/*
 * Shader specialization for a material with a BRDF (analytical or data-driven).
 * See brdf_host.h for more details on the available BRDFs.
 */
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
	void load_data(MaterialHost &object) override { mBRDF->load(object); }

private:
    BRDFShader() : Shader() {}

    // FIXME proper duplication.
    std::unique_ptr<BRDF> mBRDF;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive, const std::uint32_t version)
	{
        archive(cereal::make_nvp("class", cereal::base_class<Shader>(this)), cereal::make_nvp("brdf",mBRDF));
	}
};

CEREAL_CLASS_VERSION(BRDFShader,0)
CEREAL_REGISTER_TYPE(BRDFShader)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Shader, BRDFShader)
