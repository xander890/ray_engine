#include "environment_map_background.h"
#include "optix_device_utils.h"
#include "folders.h"
#include "logger.h"
#include "image_loader.h"
#include "optix_host_utils.h"
#include "file_dialogs.h"
#pragma warning(disable:4996) 

void EnvironmentMap::init()
{
    MissProgram::init();
	mContext["envmap_enabled"]->setInt(1);

    if(mEnvmapPath != "")
        mEnvmapTexture = loadTexture(mContext->getContext(), mEnvmapPath, optix::make_float4(1.0f));

    mProperties.environment_map_tex_id = mEnvmapTexture->get_id();
    std::string ptx_path = Folders::get_path_to_ptx("env_cameras.cu");

    auto texture_width = mEnvmapTexture->get_width();
    auto texture_height = mEnvmapTexture->get_height();

    mSamplingProperties.env_luminance = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    mSamplingProperties.marginal_f = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    mSamplingProperties.marginal_pdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    mSamplingProperties.conditional_pdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    mSamplingProperties.marginal_cdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    mSamplingProperties.conditional_cdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    
	optix::Program ray_gen_program_1 = mContext->createProgramFromPTXFile(ptx_path, "env_luminance_camera");
	mEntryPoint1 = add_entry_point(mContext, ray_gen_program_1);

	optix::Program ray_gen_program_2 = mContext->createProgramFromPTXFile(ptx_path, "env_marginal_camera");
	mEntryPoint2 = add_entry_point(mContext, ray_gen_program_2);
	
	optix::Program ray_gen_program_3 = mContext->createProgramFromPTXFile(ptx_path, "env_pdf_camera");
	mEntryPoint3 = add_entry_point(mContext, ray_gen_program_3);


	mContext["envmap_properties"]->setUserData(sizeof(EnvmapProperties), &mProperties);
	mContext["envmap_importance_sampling"]->setUserData(sizeof(EnvmapImportanceSamplingData), &mSamplingProperties);
}

void EnvironmentMap::load()
{
    MissProgram::load();

    mContext["envmap_properties"]->setUserData(sizeof(EnvmapProperties), &mProperties);
    mContext["envmap_importance_sampling"]->setUserData(sizeof(EnvmapImportanceSamplingData), &mSamplingProperties);

    if (mResample)
        presample_environment_map();
}

bool EnvironmentMap::on_draw()
{
	bool changed = false;
	if (ImmediateGUIDraw::TreeNode("Environment map"))
	{
        mEnvmapTexture->on_draw();
        ImmediateGUIDraw::Text("Path: %s", mEnvmapPath.c_str());

		changed |= ImmediateGUIDraw::InputFloat3("Multiplier##Envmapmultiplier", (float*)&mProperties.lightmap_multiplier);
		changed |= ImmediateGUIDraw::Checkbox("Importance Sample##Importancesampleenvmap", (bool*)&mProperties.importance_sample_envmap);

		static optix::float3 deltas;
		if (ImmediateGUIDraw::DragFloat3("Rotation##EnvmapDeltas", (float*)&deltas, 1.0f, -180.0f, 180.0f))
		{
			changed = true;
			mEnvmapRotationDeltas = deltas / 180.0f * M_PIf;
			mResample = true;
		}
		ImGui::TreePop();

	}
	return changed; 
}


bool EnvironmentMap::get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program)
{
    const char * prog = ray_type ==  RayType::SHADOW ? "miss_shadow" : "miss";
    program = ctx->createProgramFromPTXFile(Folders::get_path_to_ptx("environment_map_background.cu"), prog);
    return true;
}

optix::Matrix3x3 get_offset_lightmap_rotation_matrix(float delta_x, float delta_y, float delta_z, const optix::Matrix3x3& current_matrix)
{
	optix::Matrix3x3 matrix = optix::rotation_matrix3x3(optix::ZAXIS, delta_z) * optix::rotation_matrix3x3(optix::YAXIS, delta_y) * optix::rotation_matrix3x3(optix::XAXIS, delta_x);
    matrix = matrix * current_matrix;
    return matrix;
}

void EnvironmentMap::presample_environment_map()
{
    mProperties.lightmap_rotation_matrix = get_offset_lightmap_rotation_matrix(mEnvmapRotationDeltas.x, mEnvmapRotationDeltas.y, mEnvmapRotationDeltas.z, mProperties.lightmap_rotation_matrix);
    // Environment importance sampling pre-pass
    auto texture_width = mEnvmapTexture->get_width();
    auto texture_height = mEnvmapTexture->get_height();

    if (mEnvmapTexture.get() != nullptr)
    {
        Logger::info << "Presampling envmaps... (size " << std::to_string(texture_width) << " " << std::to_string(texture_height) << ")" << std::endl;
        mContext->launch(mEntryPoint1, texture_width, texture_height);
        Logger::info << "Step 1 complete." << std::endl;
        mContext->launch(mEntryPoint2, texture_width, texture_height);
        Logger::info << "Step 2 complete." << std::endl;
        mContext->launch(mEntryPoint3, texture_width, texture_height);
        Logger::info << "Step 3 complete." << std::endl;
        mResample = false;
        Logger::info << "Done." << std::endl;
    }

}

void EnvironmentMap::load(cereal::XMLInputArchiveOptix& archive)
{
    optix::Context ctx = archive.get_context();
    mContext = ctx;
    archive(
        cereal::virtual_base_class<MissProgram>(this),
        cereal::make_nvp("texture", mEnvmapTexture),
        cereal::make_nvp("delta_rotation", mEnvmapRotationDeltas),
        cereal::make_nvp("light_multiplier", mProperties.lightmap_multiplier),
        cereal::make_nvp("importance_sample", mProperties.importance_sample_envmap)
    );
}

void EnvironmentMap::save(cereal::XMLOutputArchiveOptix& archive) const
{
    archive(
        cereal::virtual_base_class<MissProgram>(this),
        cereal::make_nvp("texture", mEnvmapTexture),
        cereal::make_nvp("delta_rotation", mEnvmapRotationDeltas),
        cereal::make_nvp("light_multiplier", mProperties.lightmap_multiplier),
        cereal::make_nvp("importance_sample", mProperties.importance_sample_envmap)
    );
}

EnvironmentMap::EnvironmentMap(optix::Context & ctx, std::string envmap_file) : MissProgram(ctx), mEnvmapPath(envmap_file), mEntryPoint1(0), mEntryPoint2(0), mEntryPoint3(0)
{
	mContext = ctx;
    mEnvmapRotationDeltas = optix::make_float3(0);
    mProperties.lightmap_multiplier = optix::make_float3(1.0f);
    mProperties.importance_sample_envmap = 1;
}

EnvironmentMap::~EnvironmentMap()
{
    mEnvmapTexture.reset();
}
