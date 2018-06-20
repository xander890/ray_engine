#pragma once
#include <shader.h>
#include <mesh_host.h>
#include <bssrdf_host.h>

/*
 * Utility shader to visualize the shape of a BSSRDF on a plane.
 * Does NOT actually render the BSSRDF, this is only for debug purposes.  
 */
class BSSRDFPlaneRenderer : public Shader
{
public:
	BSSRDFPlaneRenderer(const ShaderInfo& shader_info, int camera_width, int camera_height);
	BSSRDFPlaneRenderer(BSSRDFPlaneRenderer & copy);

	void initialize_shader(optix::Context) override;
	void initialize_material(MaterialHost &object) override;
	void pre_trace_mesh(Object& object) override;
	void post_trace_mesh(Object& object) override;
	bool on_draw() override;
	Shader* clone() override { return new BSSRDFPlaneRenderer(*this); }
	void load_data(MaterialHost &mat) override;

protected:
	virtual void reset();

	int mCameraWidth;
	int mCameraHeight;

	// Gui variables
	int mShowFalseColors = 1;
	int mAngle = 45;
	float mMult = 1.0f;
	unsigned int mChannel = 0;

	static int entry_point_output;
	optix::Buffer mParameters;
	std::unique_ptr<BSSRDF> mBSSRDF;

private:
	// Serialization
	BSSRDFPlaneRenderer() : Shader() {}
	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(cereal::base_class<Shader>(this));
	}

};

CEREAL_CLASS_VERSION(BSSRDFPlaneRenderer, 0)
CEREAL_REGISTER_TYPE(BSSRDFPlaneRenderer)