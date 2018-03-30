#include "bssrdf_creator.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include <sstream>
#include <algorithm>
#include <parserstringhelpers.h>

int BSSRDFRenderer::mGlobalId = 0;

void BSSRDFRenderer::set_geometry_parameters(float theta_i, optix::float2 r,optix::float2 theta_s)
{
	mThetai = theta_i;
	mThetas = theta_s;
	mRadius = r;
	reset();
}

void BSSRDFRenderer::load_data()
{
	if (!mInitialized)
		init();
	context["ref_frame_number"]->setUint(mRenderedFrames);
    fill_geometry_data();
	context["reference_bssrdf_data"]->setUserData(sizeof(BSSRDFRendererData), &mGeometryData);
	context["reference_bssrdf_rel_ior"]->setFloat(mIor);
	context["reference_bssrdf_output_shape"]->setUserData(sizeof(OutputShape::Type), &mOutputShape);
}

void BSSRDFRenderer::set_material_parameters(float albedo, float extinction, float g, float eta)
{
	mAlbedo = albedo;
	mExtinction = extinction;
	mAsymmetry = g;
	mIor = eta;
	reset();
}

void BSSRDFRenderer::reset()
{
	if (!mInitialized)
		init();
	clear_buffer(mBSSRDFBuffer);
	clear_buffer(mBSSRDFBufferIntermediate);
	mRenderedFrames = 0;
	ScatteringMaterialProperties c;
	fill_scattering_parameters_alternative(c, 1, mIor, optix::make_float3(mAlbedo), optix::make_float3(mExtinction), optix::make_float3(mAsymmetry));
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	*cc = c;
	mProperties->unmap();
    fill_geometry_data();
}

void BSSRDFRenderer::init()
{
	mInitialized = true;
	if (mBSSRDFBufferIntermediate.get() == nullptr)
	{
		mBSSRDFBufferIntermediate = create_buffer<float>(context, RT_BUFFER_INPUT_OUTPUT, mShapeSize.x*mShapeSize.y);
		mBSSRDFBufferIntermediate->setFormat(RT_FORMAT_FLOAT);
		mBSSRDFBufferIntermediate->setSize(mShapeSize.x, mShapeSize.y);
	}

	if (mBSSRDFBuffer.get() == nullptr)
	{
		mBSSRDFBuffer = create_buffer<float>(context, RT_BUFFER_INPUT_OUTPUT, mShapeSize.x*mShapeSize.y);
		mBSSRDFBuffer->setFormat(RT_FORMAT_FLOAT);
		mBSSRDFBuffer->setSize(mShapeSize.x, mShapeSize.y);
	}

	if (mProperties.get() == nullptr)
	{
		mProperties = context->createBuffer(RT_BUFFER_INPUT);
		mProperties->setFormat(RT_FORMAT_USER);
		mProperties->setElementSize(sizeof(ScatteringMaterialProperties));
		mProperties->setSize(1);
	}

	if (mSolidAngleBuffer.get() == nullptr)
	{
		mSolidAngleBuffer = context->createBuffer(RT_BUFFER_INPUT);
		mSolidAngleBuffer->setFormat(RT_FORMAT_FLOAT);
		mSolidAngleBuffer->setSize(mShapeSize.x, mShapeSize.y);
		fill_solid_angle_buffer();
	}

	reset();
}

bool BSSRDFRenderer::on_draw(unsigned int flags)
{
	if(flags == HIDE_ALL)
		return false;

	bool changed = false;

	float backup_theta_i = mThetai;
	optix::float2 backup_theta_s = mThetas;
	optix::float2 backup_r = mRadius;

	changed |= ImmediateGUIDraw::SliderFloat(GUI_LABEL("Incoming light angle (deg.)", mId), &backup_theta_i, 0, 90);
	if ((flags & SHOW_GEOMETRY != 0))
	{
		changed |= ImmediateGUIDraw::InputFloat(GUI_LABEL("Radius (lower)", mId), &backup_r.x, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat(GUI_LABEL("Radius (upper)", mId), &backup_r.y, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::SliderFloat(GUI_LABEL("Angle on plane (lower)", mId), &backup_theta_s.x, 0, 360);
		changed |= ImmediateGUIDraw::SliderFloat(GUI_LABEL("Angle on plane (upper)", mId), &backup_theta_s.y, 0, 360);
	}

	if(changed && !mIsReadOnly)
	{
		set_geometry_parameters(backup_theta_i, backup_r, backup_theta_s);
	}

	float albedo = mAlbedo, ext = mExtinction, g = mAsymmetry, eta = mIor;
	if (flags & SHOW_MATERIAL != 0)
	{
		changed |= ImmediateGUIDraw::InputFloat(GUI_LABEL("Albedo", mId), &albedo, 0, 0, -1, mIsReadOnly? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat(GUI_LABEL("Extinction", mId), &ext, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat(GUI_LABEL("G", mId), &g, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat(GUI_LABEL("Relative IOR", mId), &eta, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
	}

	if(changed && !mIsReadOnly)
	{
		set_material_parameters(albedo, ext, g, eta);
	}

	if (changed)
	{
		reset();
	}

	return changed;
}

void BSSRDFRenderer::set_shape(OutputShape::Type shape)
{
	mOutputShape = shape;
	mShapeSize = default_size(shape);
	mBSSRDFBufferIntermediate->setSize(mShapeSize.x, mShapeSize.y);
	mBSSRDFBuffer->setSize(mShapeSize.x, mShapeSize.y);
}

void BSSRDFRenderer::fill_geometry_data() {
	optix::float2 theta_s = deg2rad(mThetas);
    mGeometryData.mThetai = deg2rad(mThetai);
    mGeometryData.mThetas = theta_s;
    mGeometryData.mRadius = mRadius;
	mGeometryData.mArea = (mRadius.y*mRadius.y - mRadius.x*mRadius.x) * 0.5f * (theta_s.y - theta_s.x);
	mGeometryData.mSolidAngleBuffer = mSolidAngleBuffer->getId();
	mGeometryData.mDeltaR = mRadius.y - mRadius.x;
	mGeometryData.mDeltaThetas = theta_s.y - theta_s.x;
}

void BSSRDFRenderer::get_geometry_parameters(float &theta_i, optix::float2 &r, optix::float2 &theta_s) {
    theta_i = mThetai;
    r = mRadius;
    theta_s = mThetas;
}

void BSSRDFRenderer::get_material_parameters(float &albedo, float &extinction, float &g, float &eta) {
    albedo = mAlbedo;
    extinction = mExtinction;
    g = mAsymmetry;
    eta = mIor;
}

BSSRDFRenderer::BSSRDFRenderer(optix::Context &ctx, const OutputShape::Type shape, const optix::int2 &shape_size) : context(ctx) {
	mOutputShape = shape;
	mId = mGlobalId++;
	if(shape_size.x > -1 && shape_size.y > -1)
	{
		mShapeSize = optix::make_uint2(shape_size.x, shape_size.y);
	}
	else
	{
		mShapeSize = default_size(shape);
	}
}

BSSRDFRenderer::~BSSRDFRenderer() {
	mBSSRDFBuffer->destroy();
	mBSSRDFBufferIntermediate->destroy();
}

optix::uint2 BSSRDFRenderer::default_size(OutputShape::Type shape) {
	return shape == OutputShape::HEMISPHERE ? optix::make_uint2(160,40) : optix::make_uint2(80,80);
}

void BSSRDFRenderer::fill_solid_angle_buffer()
{
	float * buf = reinterpret_cast<float*>(mSolidAngleBuffer->map());
	for(int i = 0; i < mShapeSize.x*mShapeSize.y; i++)
	{
		buf[i] = 2.0f * M_PIf / (mShapeSize.x * mShapeSize.y);
	}
	mSolidAngleBuffer->unmap();
}

void BSSRDFRendererModel::init()
{
	BSSRDFRenderer::init();
	std::string ptx_path = get_path_ptx("planar_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["planar_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);
	BufPtr2D<float> ptr = BufPtr2D<float>(mBSSRDFBufferIntermediate->getId());
	context["planar_resulting_flux_intermediate"]->setUserData(sizeof(BufPtr2D<float>), &ptr);
	BufPtr2D<float> ptr2 = BufPtr2D<float>(mBSSRDFBuffer->getId());
	context["planar_resulting_flux"]->setUserData(sizeof(BufPtr2D<float>), &ptr2);

}

void BSSRDFRendererModel::set_dipole(ScatteringDipole::Type dipole)
{
	mBSSRDF.reset();
	mBSSRDF = BSSRDF::create(context, dipole);
}

void BSSRDFRendererModel::render()
{
	if (!mInitialized)
		init();
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(20, 20, -1);
	context->launch(entry_point, mShapeSize.x, mShapeSize.y);
	context->launch(entry_point_post, mShapeSize.x, mShapeSize.y);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;
}

bool BSSRDFRendererModel::on_draw(unsigned int flags)
{
	if (BSSRDFRenderer::on_draw(flags))
		reset();
	mBSSRDF->on_draw();
	return false;
}

void BSSRDFRendererModel::load_data()
{
	BSSRDFRenderer::load_data();
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	mBSSRDF->load(1.1f, *cc);
	auto type = mBSSRDF->get_type();
	context["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &type);


	mProperties->unmap();
}

void BSSRDFRendererSimulated::init()
{
	BSSRDFRenderer::init();
	std::string ptx_path = get_path_ptx("reference_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["reference_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);
	BufPtr2D<float> ptr = BufPtr2D<float>(mBSSRDFBufferIntermediate->getId());
	context["reference_resulting_flux_intermediate"]->setUserData(sizeof(BufPtr2D<float>), &ptr);
	BufPtr2D<float> ptr2 = BufPtr2D<float>(mBSSRDFBuffer->getId());
	context["reference_resulting_flux"]->setUserData(sizeof(BufPtr2D<float>), &ptr2);

    float scattering = mAlbedo * mExtinction;
    set_bias(1.0f/(3.0f * scattering));
}

void BSSRDFRendererSimulated::render()
{
	if (!mInitialized)
		init();
	context->launch(entry_point, mSamples);
	context->launch(entry_point_post, mShapeSize.x, mShapeSize.y);
	mRenderedFrames++;
}

void BSSRDFRendererSimulated::load_data()
{
	BSSRDFRenderer::load_data();
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);

    if(mbUseAutomaticBias)
    {
        float bias_mfp = (mRadius.x + mRadius.y) / 2.0f;
        bias_mfp = fminf(bias_mfp, 1.0f);
        mBssrdfOptions.mBias = 1.0f / (bias_mfp * bias_mfp);
        mBiasInMfps = bias_mfp;
    }

    context["reference_bssrdf_simulated_options"]->setUserData(sizeof(BSSRDFSimulatedOptions), &mBssrdfOptions);
}

bool BSSRDFRendererSimulated::on_draw(unsigned int flags)
{
	std::stringstream ss;
	ss << "Rendered: " << mRenderedFrames << " frames, " << mRenderedFrames*mSamples << " samples" << std::endl;
	ImmediateGUIDraw::Text("%s",ss.str().c_str());
	bool changed = BSSRDFRenderer::on_draw(flags);

	static int smpl = mSamples;
	if (ImmediateGUIDraw::InputInt(GUI_LABEL("Samples", mId), &smpl, 1, 100, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0))
	{
		set_samples(smpl);
	}

	static int iter = mMaxIterations;
	if (ImmediateGUIDraw::InputInt(GUI_LABEL("Maximum iterations", mId), &iter, 1, 100, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0))
	{
		set_max_iterations(iter);
	}

    std::string s = IntegrationMethod::get_full_string();
	std::replace(s.begin(), s.end(), ' ', '\0');

	changed |= ImmediateGUIDraw::Combo(GUI_LABEL("Integration method", mId), (int*)&mBssrdfOptions.mIntegrationMethod, s.c_str(), IntegrationMethod::count());
	changed |= ImmediateGUIDraw::Checkbox(GUI_LABEL("Automatic bias", mId), &mbUseAutomaticBias);

    if (ImmediateGUIDraw::InputFloat(GUI_LABEL("Bias reduction bound", mId), &mBiasInMfps, mIsReadOnly || mbUseAutomaticBias? ImGuiInputTextFlags_ReadOnly : 0))
    {
        changed = true;
        set_bias(mBiasInMfps);
    }

	std::string s2 = BiasMode::get_full_string();
	std::vector<std::string> tokens;
	split(tokens, s2, ' ');
	std::vector<const char*> c;
	for(std::string & s : tokens)
	{
		c.push_back(s.c_str());
	}
	if((flags & SHOW_EXTRA_OPTIONS) != 0)
		changed |= ImmediateGUIDraw::Combo(GUI_LABEL("Bias rendering type", mId), (int*)&mBssrdfOptions.mbBiasMode, &c[0], BiasMode::count(), BiasMode::count());

	if((flags & SHOW_EXTRA_OPTIONS) != 0)
		changed |= ImmediateGUIDraw::Checkbox(GUI_LABEL("Cosine weighted", mId), &mBssrdfOptions.mbCosineWeighted);

	if (changed)
		reset();
	return changed;
}

void BSSRDFRendererSimulated::set_samples(int samples)
{
	mSamples = samples;
	reset();
}

void BSSRDFRendererSimulated::set_max_iterations(int max_iter)
{
	mMaxIterations = max_iter;
	reset();
}


void BSSRDFRendererSimulated::set_integration_method(IntegrationMethod::Type method) {
    mBssrdfOptions.mIntegrationMethod = method;
    reset();
}

void BSSRDFRendererSimulated::set_bias_visualization_method(BiasMode::Type biasmethod) {
    mBssrdfOptions.mbBiasMode = biasmethod;
    reset();
}

void BSSRDFRendererSimulated::set_bias(float bias) {
    mBssrdfOptions.mBias = 1.0f / (bias * bias);
	mBiasInMfps = bias;
    reset();
}

void BSSRDFRendererSimulated::set_material_parameters(float albedo, float extinction, float g, float eta) {
    BSSRDFRenderer::set_material_parameters(albedo, extinction, g, eta);
}

void BSSRDFRendererSimulated::set_cosine_weighted(bool cosine_weighted) {
	mBssrdfOptions.mbCosineWeighted = cosine_weighted;
}
