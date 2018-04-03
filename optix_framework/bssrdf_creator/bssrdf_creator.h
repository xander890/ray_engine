#pragma once
#include <string>
#include <host_device_common.h>
#include <scattering_properties.h>
#include <full_bssrdf_host_device_common.h>
#include <bssrdf_properties.h>
#include <bssrdf_host.h>
#include "photon_trace_structs.h"


#define hash "#"
#define GUI_LABEL_IMPL(label, id) (std::string(label hash hash label) + std::to_string(id)).c_str()
#define GUI_LABEL(label, id) GUI_LABEL_IMPL(label, id)

class BSSRDFRenderer
{
public:

	BSSRDFRenderer(optix::Context & ctx, const OutputShape::Type shape = DEFAULT_SHAPE, const optix::int2 & shape_size = optix::make_int2(-1));
    virtual ~BSSRDFRenderer();

	virtual void load_data();
	virtual void set_geometry_parameters(float theta_i, optix::float2 r, optix::float2 theta_s);
	virtual void set_material_parameters(float albedo, float extinction, float g, float eta);
    void get_geometry_parameters(float &theta_i, optix::float2 &r, optix::float2& theta_s);
    void get_material_parameters(float &albedo, float &extinction, float& g, float& eta);

	virtual void reset();
	virtual void init();
	virtual void render() = 0;

	virtual optix::Buffer get_output_buffer() { return mBSSRDFBuffer; }
	optix::uint2 get_size() { return mShapeSize; }
	size_t get_storage_size() { return mShapeSize.x*mShapeSize.y; }
	virtual size_t get_samples() { return 1; }

	virtual bool on_draw(unsigned int flags);

	void set_read_only(bool is_read_only) { mIsReadOnly = is_read_only;  }
	void set_shape(OutputShape::Type shape);
	OutputShape::Type get_shape() { return mOutputShape; }

	enum GUIFlags
	{
		SHOW_MATERIAL = 0x01,
		SHOW_GEOMETRY = 0x02,
		SHOW_EXTRA_OPTIONS = 0x04,
		SHOW_ALL = 0x07,
		HIDE_ALL = 0x00,
	};

protected:
	void fill_solid_angle_buffer();
    void fill_geometry_data();
    static optix::uint2 default_size(OutputShape::Type shape);

	OutputShape::Type mOutputShape;
	int entry_point = -1;
	int entry_point_post = -1;
    unsigned int mRenderedFrames = 0;

	optix::uint2 mShapeSize;
	optix::Context context = nullptr;

	optix::Buffer mBSSRDFBufferIntermediate = nullptr;
	optix::Buffer mBSSRDFBuffer = nullptr;
    optix::Buffer mProperties = nullptr;
	optix::Buffer mSolidAngleBuffer = nullptr;

	// Geometric properties
    BSSRDFRendererData mGeometryData;

	float mThetai = 0.0f;
	optix::float2 mThetas = optix::make_float2(0.0f, 7.5f);
    optix::float2 mRadius = optix::make_float2(0.0f, 1.0f);
	// Scattering properties

	float mAlbedo = 0.8f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.0f;
    float mIor = 1.f;

    bool mIsReadOnly = false;
    bool mInitialized = false;
    static int mGlobalId;
    int mId;
};

class BSSRDFRendererSimulated : public BSSRDFRenderer
{
public:
	BSSRDFRendererSimulated(optix::Context & ctx, const OutputShape::Type shape = DEFAULT_SHAPE, const optix::int2 & shape_size = optix::make_int2(-1), const unsigned int samples = (int)1e9) : BSSRDFRenderer(ctx, shape, shape_size), mSamples(samples)
	{
	}

	void init() override;
	void render() override;
	void load_data() override;
	bool on_draw(unsigned int flags) override;
	virtual void set_samples(int samples);
	virtual void set_max_iterations(int max_iter);

    void set_material_parameters(float albedo, float extinction, float g, float eta) override;

	size_t get_samples() override { return mSamples * mRenderedFrames; }
    float get_bias() {return mBiasInMfps; }

    void set_integration_method(IntegrationMethod::Type method);
    void set_bias_visualization_method(BiasMode::Type biasmethod);
    void set_bias(float bias);
    void set_cosine_weighted(bool cosine_weighted);

protected:
	unsigned int mSamples;
	unsigned int mMaxIterations = (int)1e9;
    BSSRDFSimulatedOptions mBssrdfOptions;
	float mBiasInMfps;
    bool mbUseAutomaticBias = true;
};

class BSSRDFRendererModel : public BSSRDFRenderer
{
public:
	BSSRDFRendererModel(optix::Context & ctx, const OutputShape::Type shape = DEFAULT_SHAPE, const optix::int2 & shape_size = optix::make_int2(-1), const ScatteringDipole::Type & dipole = ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF) : BSSRDFRenderer(ctx, shape, shape_size)
	{
		mBSSRDF = std::move(BSSRDF::create(ctx, dipole));
	}

	void init() override;
	void render() override;
	bool on_draw(unsigned int flags) override;
	void load_data() override;
	void set_dipole(ScatteringDipole::Type dip);
	ScatteringDipole::Type get_dipole() {return mBSSRDF->get_type();}
	std::unique_ptr<BSSRDF> mBSSRDF = nullptr;
};