#ifndef glossy_h__
#define glossy_h__
#include "shader.h"

struct MERLBrdf
{
    std::vector<float> data;
    optix::float3 reflectance;
    std::string name;
};

class GlossyShader : public Shader
{
public:
    virtual ~GlossyShader() = default;
    GlossyShader() : Shader() {}
	GlossyShader(const GlossyShader &) = default;

    void initialize_shader(optix::Context context, const ShaderInfo& shader_info) override;
    
    void load_into_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override;
	virtual Shader* clone() override { return new GlossyShader(*this); }

private:
    void set_data(Mesh& object);
	// FIXME proper struct & copy constructor
	float blinn_exponent;
    float2 anisotropic_exp;
    float3 x_axis_anisotropic;

	// FIXME move me somewhere elses
	std::vector<std::string> brdf_names;
	std::string merl_folder;
	std::map<std::string, MERLBrdf> merl_database;
    float3 merl_correction;
    bool use_merl_brdf;
};


#endif // glossy_h__
