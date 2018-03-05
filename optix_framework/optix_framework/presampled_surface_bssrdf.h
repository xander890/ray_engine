#ifndef PRESAMPLED_H
#define PRESAMPLED_H

#include <shader.h>
#include <mesh.h>
#include <bssrdf_host.h>

class PresampledSurfaceBssrdf : public Shader 
{
public:
    PresampledSurfaceBssrdf(const ShaderInfo& shader_info) : Shader(shader_info), entry_point(0) { }
	PresampledSurfaceBssrdf(PresampledSurfaceBssrdf& copy);

    void initialize_shader(optix::Context) override;
    void initialize_mesh(Mesh& object) override;
    void pre_trace_mesh(Mesh& object) override;
	void load_data(Mesh & object) override;
	bool on_draw() override;
	virtual Shader* clone() override { return new PresampledSurfaceBssrdf(*this); }

private:
    int entry_point;
	optix::Buffer mSampleBuffer;
	unsigned int mSamples = 1000;
	float mArea;
	optix::Buffer mCdfBuffer;
	std::unique_ptr<BSSRDF> mBSSRDF;
	bool mExcludeBackFaces = false;
};



#endif // PRESAMPLED_H
