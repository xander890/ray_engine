#include "bssrdf_creator.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"
#include "folders.h"
#include <sstream>

void BSSRDFRenderer::set_geometry_parameters(float theta_i, float r, float theta_s)
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
	context["reference_bssrdf_theta_i"]->setFloat(deg2rad(mThetai));
	context["reference_bssrdf_theta_s"]->setFloat(deg2rad(mThetas));
	context["reference_bssrdf_radius"]->setFloat(mRadius);
	context["reference_bssrdf_rel_ior"]->setFloat(mIor);
	context["reference_bssrdf_output_shape"]->setInt(static_cast<int>(mOutputShape));
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

	reset();
}

bool BSSRDFRenderer::on_draw(bool show_material_params)
{
	bool changed = false;

	float backup_theta_i = mThetai;
	float backup_theta_s = mThetas;

	changed |= ImmediateGUIDraw::SliderFloat("Incoming light angle (deg.)", &mThetai, 0, 90);
	if (mOutputShape == HEMISPHERE)
	{
		changed |= ImmediateGUIDraw::InputFloat("Radius", &mRadius, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::SliderFloat("Angle on plane", &mThetas, 0, 360);
	}
	mThetai = mIsReadOnly ? backup_theta_i : mThetai;
	mThetas = mIsReadOnly ? backup_theta_s : mThetas;

	if (show_material_params)
	{
		changed |= ImmediateGUIDraw::InputFloat("Albedo##RefAlbedo", &mAlbedo, 0, 0, -1, mIsReadOnly? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat("Extinction##RefExtinction", &mExtinction, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat("G##RefAsymmetry", &mAsymmetry, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat("Relative IOR##RefRelIOR", &mIor, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
	}

	if (changed)
	{
		reset();

	}

	return changed;
}

void BSSRDFRenderer::set_shape(OutputShape shape)
{
	mOutputShape = shape;
	mShapeSize = default_size(shape);
	//resize_glbo_buffer<float>(mBSSRDFBufferIntermediate, mShapeSize.x*mShapeSize.y);
	//resize_glbo_buffer<float>(mBSSRDFBuffer, mShapeSize.x*mShapeSize.y);

	mBSSRDFBufferIntermediate->setSize(mShapeSize.x, mShapeSize.y);
	mBSSRDFBuffer->setSize(mShapeSize.x, mShapeSize.y);
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

bool BSSRDFRendererModel::on_draw(bool show_material_params)
{
	if (BSSRDFRenderer::on_draw(show_material_params))
		reset();
	return false;
}

void BSSRDFRendererModel::load_data()
{
	BSSRDFRenderer::load_data();
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	mBSSRDF->load(*cc);
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

}

bool BSSRDFRendererSimulated::on_draw(bool show_material_params)
{
	std::stringstream ss;
	ss << "Rendered: " << mRenderedFrames << " frames, " << mRenderedFrames*mSamples << " samples" << std::endl;
	ImmediateGUIDraw::Text(ss.str().c_str());
	bool changed = BSSRDFRenderer::on_draw(show_material_params);

	static int smpl = mSamples;
	if (ImmediateGUIDraw::InputInt("Samples", (int*)&smpl, 1, 100, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0))
	{
		set_samples(smpl);
	}

	static int iter = mMaxIterations;
	if (ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&iter, 1, 100, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0))
	{
		set_max_iterations(iter);
	}

	if (changed)
		reset();
	return false;
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
