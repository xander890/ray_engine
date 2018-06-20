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

    if(envmap_path != "")
        environment_sampler = loadTexture(mContext->getContext(), envmap_path, optix::make_float4(1.0f));

    properties.environment_map_tex_id = environment_sampler->get_id();
    std::string ptx_path = get_path_ptx("env_cameras.cu");

    auto texture_width = environment_sampler->get_width();
    auto texture_height = environment_sampler->get_height();

    sampling_properties.env_luminance = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_f = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.marginal_pdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_pdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_cdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_cdf = (mContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    
	optix::Program ray_gen_program_1 = mContext->createProgramFromPTXFile(ptx_path, "env_luminance_camera");
	camera_1 = add_entry_point(mContext, ray_gen_program_1);

	optix::Program ray_gen_program_2 = mContext->createProgramFromPTXFile(ptx_path, "env_marginal_camera");
	camera_2 = add_entry_point(mContext, ray_gen_program_2);
	
	optix::Program ray_gen_program_3 = mContext->createProgramFromPTXFile(ptx_path, "env_pdf_camera");
	camera_3 = add_entry_point(mContext, ray_gen_program_3);

    property_buffer = create_and_initialize_buffer<EnvmapProperties>(mContext, properties);
    sampling_property_buffer = create_and_initialize_buffer<EnvmapImportanceSamplingData>(mContext, sampling_properties);

    BufPtr<EnvmapProperties> b = BufPtr<EnvmapProperties>(property_buffer->getId());
	mContext["envmap_properties"]->setUserData(sizeof(BufPtr<EnvmapProperties>), &b);

    BufPtr<EnvmapImportanceSamplingData> b2 = BufPtr<EnvmapImportanceSamplingData>(sampling_property_buffer->getId());
	mContext["envmap_importance_sampling"]->setUserData(sizeof(BufPtr<EnvmapImportanceSamplingData>), &b2);
}

void EnvironmentMap::load()
{
    MissProgram::load();
    void * ptr = property_buffer->map();
    memcpy(ptr, &properties, sizeof(EnvmapProperties));
    property_buffer->unmap();

    if (resample_envmaps)
        presample_environment_map();
}

bool EnvironmentMap::on_draw()
{
	bool changed = false;
	if (ImmediateGUIDraw::TreeNode("Environment map"))
	{
        environment_sampler->on_draw();
        ImmediateGUIDraw::Text("Path: %s", envmap_path.c_str());

		changed |= ImmediateGUIDraw::InputFloat3("Multiplier##Envmapmultiplier", (float*)&properties.lightmap_multiplier);
		changed |= ImmediateGUIDraw::Checkbox("Importance Sample##Importancesampleenvmap", (bool*)&properties.importance_sample_envmap);

		static optix::float3 deltas;
		if (ImmediateGUIDraw::DragFloat3("Rotation##EnvmapDeltas", (float*)&deltas, 1.0f, -180.0f, 180.0f))
		{
			changed = true;
			envmap_deltas = deltas / 180.0f * M_PIf;
			resample_envmaps = true;
		}
		ImGui::TreePop();

	}
	return changed; 
}


bool EnvironmentMap::get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program)
{
    const char * prog = ray_type ==  RayType::SHADOW ? "miss_shadow" : "miss";
    program = ctx->createProgramFromPTXFile(get_path_ptx("environment_map_background.cu"), prog);
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
    properties.lightmap_rotation_matrix = get_offset_lightmap_rotation_matrix(envmap_deltas.x, envmap_deltas.y, envmap_deltas.z, properties.lightmap_rotation_matrix);
    // Environment importance sampling pre-pass
    auto texture_width = environment_sampler->get_width();
    auto texture_height = environment_sampler->get_height();

    if (environment_sampler.get() != nullptr)
    {
        Logger::info << "Presampling envmaps... (size " << std::to_string(texture_width) << " " << std::to_string(texture_height) << ")" << std::endl;
        mContext->launch(camera_1, texture_width, texture_height);
        Logger::info << "Step 1 complete." << std::endl;
        mContext->launch(camera_2, texture_width, texture_height);
        Logger::info << "Step 2 complete." << std::endl;
        mContext->launch(camera_3, texture_width, texture_height);
        Logger::info << "Step 3 complete." << std::endl;
        resample_envmaps = false;
        Logger::info << "Done." << std::endl;
    }

}

EnvironmentMap::EnvironmentMap(optix::Context & ctx, std::string envmap_file) : MissProgram(ctx), envmap_path(envmap_file), camera_1(0), camera_2(0), camera_3(0)
{
    envmap_deltas = optix::make_float3(0);
    properties.lightmap_multiplier = optix::make_float3(1.0f);
    properties.importance_sample_envmap = 1;
}

EnvironmentMap::~EnvironmentMap()
{
    environment_sampler.reset();
    property_buffer->destroy();
    sampling_property_buffer->destroy();
}
