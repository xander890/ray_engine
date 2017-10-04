#ifndef PRESAMPLED_H
#define PRESAMPLED_H

#include <shader.h>
#include <mesh.h>

class PresampledSurfaceBssrdf : public Shader 
{
public:
    PresampledSurfaceBssrdf(const ShaderInfo& shader_info) : Shader(shader_info), entry_point(0) { }

    void initialize_shader(optix::Context) override;
    void initialize_mesh(Mesh& object) override;
    void pre_trace_mesh(Mesh& object) override;
	bool on_draw() override;
	virtual Shader* clone() override { return new PresampledSurfaceBssrdf(*this); }

private:
    int entry_point;
	optix::Buffer mSampleBuffer;
	unsigned int mSamples = 1000;
};



#endif // PRESAMPLED_H
