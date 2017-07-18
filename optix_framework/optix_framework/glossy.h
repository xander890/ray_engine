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

    void initialize_shader(optix::Context context, int illum) override;
    
    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override;

private:
    void set_data(Mesh& object);
    float blinn_exponent;
    float2 anisotropic_exp;
    float3 x_axis_anisotropic;
    std::vector<std::string> brdf_names;
    std::string merl_folder;

    std::map<std::string, MERLBrdf> merl_database;
    float3 merl_correction;
    bool use_merl_brdf;
};


#endif // glossy_h__
