#ifndef PRESAMPLED_H
#define PRESAMPLED_H

#include <shader.h>
#include <mesh.h>
using namespace optix;

const unsigned int SAMPLES_FRAME = 1000;


class PresampledSurfaceBssrdf : public Shader 
{
public:
    PresampledSurfaceBssrdf() : Shader(), entry_point(0) { }

    void initialize_shader(optix::Context, const ShaderInfo& shader_info) override;
    void load_into_mesh(Mesh& object) override;
    void pre_trace_mesh(Mesh& object) override;
	virtual Shader* clone() override { return new PresampledSurfaceBssrdf(*this); }

private:
    int entry_point;
    Buffer m_samples;
};



#endif // PRESAMPLED_H
