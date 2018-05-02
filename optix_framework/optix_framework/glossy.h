#ifndef glossy_h__
#define glossy_h__
#include "shader.h"
#include "brdf_host.h"

struct MERLBrdf
{
    std::string name;
};

class BRDFShader : public Shader
{
public:
    virtual ~BRDFShader() = default;
    BRDFShader(const ShaderInfo& shader_info) : Shader(shader_info) {}
	BRDFShader(const BRDFShader &) = default;

    void initialize_shader(optix::Context context) override;
    
    void initialize_mesh(Object & object) override;
    void pre_trace_mesh(Object & object) override;
	virtual Shader* clone() override { return new BRDFShader(*this); }
    bool on_draw() override;

private:
    // FIXME proper duplication.
    std::shared_ptr<BRDF> mBRDF;
};


#endif // glossy_h__
