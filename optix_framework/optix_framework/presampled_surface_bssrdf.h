#pragma once
#include <shader.h>
#include <mesh.h>
#include <bssrdf_host.h>

class PresampledSurfaceBssrdf : public Shader 
{
public:
    PresampledSurfaceBssrdf(const ShaderInfo& shader_info) : Shader(shader_info), entry_point(0) { }
	PresampledSurfaceBssrdf(PresampledSurfaceBssrdf& copy);

    void initialize_shader(optix::Context) override;
    void initialize_material(MaterialHost &object) override;
    void pre_trace_mesh(Object& object) override;
	void load_data(MaterialHost &object) override;
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

	PresampledSurfaceBssrdf() : Shader() {}
	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::base_class<Shader>(this));
	}

};

CEREAL_REGISTER_TYPE(PresampledSurfaceBssrdf)
