#include "environment_map_background.h"
#include "optix_helpers.h"
#include "folders.h"
#include "logger.h"
#include "ImageLoader.h"
#include "optix_utils.h"
#include "dialogs.h"

#pragma warning(disable:4996) 
using namespace optix;
void EnvironmentMap::init(optix::Context & ctx)
{
    MissProgram::init(ctx);
    context = ctx;
    ctx["envmap_enabled"]->setInt(1);

    if(envmap_path != "")
        environment_sampler = loadTexture(context->getContext(), envmap_path, make_float4(1.0f));

    properties.environment_map_tex_id = environment_sampler->get_id();
    std::string ptx_path = get_path_ptx("env_cameras.cu");

    auto texture_width = environment_sampler->get_width();
    auto texture_height = environment_sampler->get_height();

    sampling_properties.env_luminance = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_f = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.marginal_pdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_pdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_cdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_cdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    
    Program ray_gen_program_1 = ctx->createProgramFromPTXFile(ptx_path, "env_luminance_camera");
	camera_1 = add_entry_point(ctx, ray_gen_program_1);

	Program ray_gen_program_2 = ctx->createProgramFromPTXFile(ptx_path, "env_marginal_camera");
	camera_2 = add_entry_point(ctx, ray_gen_program_2);
	
	Program ray_gen_program_3 = ctx->createProgramFromPTXFile(ptx_path, "env_pdf_camera");
	camera_3 = add_entry_point(ctx, ray_gen_program_3);

    property_buffer = create_and_initialize_buffer<EnvmapProperties>(context, properties);
    sampling_property_buffer = create_and_initialize_buffer<EnvmapImportanceSamplingData>(context, sampling_properties);

    BufPtr<EnvmapProperties> b = BufPtr<EnvmapProperties>(property_buffer->getId());
    ctx["envmap_properties"]->setUserData(sizeof(BufPtr<EnvmapProperties>), &b);

    BufPtr<EnvmapImportanceSamplingData> b2 = BufPtr<EnvmapImportanceSamplingData>(sampling_property_buffer->getId());
    ctx["envmap_importance_sampling"]->setUserData(sizeof(BufPtr<EnvmapImportanceSamplingData>), &b2);
}

void EnvironmentMap::set_into_gpu(optix::Context & ctx)
{
    MissProgram::set_into_gpu(ctx);
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

Matrix3x3 get_offset_lightmap_rotation_matrix(float delta_x, float delta_y, float delta_z, const optix::Matrix3x3& current_matrix)
{
    Matrix3x3 matrix = rotation_matrix3x3(ZAXIS, delta_z) * rotation_matrix3x3(YAXIS, delta_y) * rotation_matrix3x3(XAXIS, delta_x);
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
        context->launch(camera_1, texture_width, texture_height);
        Logger::info << "Step 1 complete." << std::endl;
        context->launch(camera_2, texture_width, texture_height);
        Logger::info << "Step 2 complete." << std::endl;
        context->launch(camera_3, texture_width, texture_height);
        Logger::info << "Step 3 complete." << std::endl;
        resample_envmaps = false;
        Logger::info << "Done." << std::endl;
    }

}

EnvironmentMap::EnvironmentMap(std::string envmap_file) : envmap_path(envmap_file),
                                                          camera_1(0), camera_2(0), camera_3(0)
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
