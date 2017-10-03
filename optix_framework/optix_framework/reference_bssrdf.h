#pragma once

#include <shader.h>
#include <mesh.h>

class ReferenceBSSRDF : public Shader
{
public:
	ReferenceBSSRDF() : Shader(), entry_point(0) { }

	void initialize_shader(optix::Context, const ShaderInfo& shader_info) override;
	void initialize_mesh(Mesh& object) override;
	void pre_trace_mesh(Mesh& object) override;
	bool on_draw() override;
	virtual Shader* clone() override { return new ReferenceBSSRDF(*this); }

private:
	int entry_point;
	optix::Buffer mBSSRDFBuffer;
	unsigned int mSamples = 1000;
	optix::uint2 mHemisphereSize = optix::make_uint2(180, 90);
};

