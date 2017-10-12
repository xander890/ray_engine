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
    GlossyShader(const ShaderInfo& shader_info) : Shader(shader_info) {}
	GlossyShader(const GlossyShader &) = default;

    void initialize_shader(optix::Context context) override;
    
    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override;
	virtual Shader* clone() override { return new GlossyShader(*this); }

private:
    void set_data(Mesh& object);
	// FIXME proper struct & copy constructor
	float blinn_exponent;
    optix::float2 anisotropic_exp;
	optix::float3 x_axis_anisotropic;

	// FIXME move me somewhere elses
	std::vector<std::string> brdf_names;
	std::string merl_folder;
	std::map<std::string, MERLBrdf> merl_database;
	optix::float3 merl_correction;
    bool use_merl_brdf;
};


#endif // glossy_h__
