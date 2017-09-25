#include "environment_map_background.h"
#include "optix_helpers.h"
#include "folders.h"
#include "host_device_common.h"
#include "environment_map.h"
#include <cmath>
#include "enums.h"
#include "SampleScene.h"
#include "ImageLoader.h"
#include "optix_utils.h"
#include "dialogs.h"

#pragma warning(disable:4996) 
using namespace optix;
void EnvironmentMap::init(optix::Context & ctx)
{
    MissProgram::init(ctx);
    context = ctx;
    camera_1 = ctx->getEntryPointCount();
    ctx->setEntryPointCount(camera_1 + 3);
    camera_2 = camera_1 + 1;
    camera_3 = camera_2 + 1;
    ctx["envmap_enabled"]->setInt(1);

    envmap_deltas = ConfigParameters::get_parameter<float3>("light", "envmap_deltas", make_float3(0), "Rotation offsetof environment map.");
    properties.lightmap_multiplier = make_float3(ConfigParameters::get_parameter<float>("light", "lightmap_multiplier", (1.0f), "Environment map multiplier"));
    bool is_env = false;
    is_env = true;
    std::string ptx_path = get_path_ptx("env_cameras.cu");
	envmap_path = Folders::texture_folder + envmap_file;
    environment_sampler = loadTexture(context->getContext(), envmap_path, make_float3(1.0f));
    properties.environment_map_tex_id = environment_sampler->getId();
    properties.importance_sample_envmap = 1;

    RTsize w, h;
    environment_sampler.get()->getBuffer(0, 0)->getSize(w, h);
    texture_width = static_cast<int>(w);
    texture_height = static_cast<int>(h);

    sampling_properties.env_luminance = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_f = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.marginal_pdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_pdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_cdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_cdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    
    Program ray_gen_program_1 = ctx->createProgramFromPTXFile(ptx_path, "env_luminance_camera");
    ctx->setRayGenerationProgram((camera_1), ray_gen_program_1);   
    Program ray_gen_program_2 = ctx->createProgramFromPTXFile(ptx_path, "env_marginal_camera");
    ctx->setRayGenerationProgram((camera_2), ray_gen_program_2);
    Program ray_gen_program_3 = ctx->createProgramFromPTXFile(ptx_path, "env_pdf_camera");
    ctx->setRayGenerationProgram((camera_3), ray_gen_program_3);    

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
	const char* env_map_correction_group = "Environment map";
	if (ImmediateGUIDraw::TreeNode("Environment map"))
	{
		char txt[512];
		envmap_path.copy(txt, envmap_path.length());
		ImmediateGUIDraw::InputText("Path", txt, 512, ImGuiInputTextFlags_ReadOnly);

		std::string filePath;
		if (ImmediateGUIDraw::Button("Load new envmap..."))
		{
			std::string filePath;
			if (Dialogs::openFileDialog(filePath))
			{
				environment_sampler->destroy();
				environment_sampler = loadTexture(context->getContext(), filePath, make_float3(1.0f));
				int id = environment_sampler->getId();
				properties.environment_map_tex_id = id;
				envmap_path = filePath;
				resample_envmaps = true;
				changed = true;
			}
		}

		changed |= ImmediateGUIDraw::InputFloat3("Multiplier##Envmapmultiplier", (float*)&properties.lightmap_multiplier);
		changed |= ImmediateGUIDraw::Checkbox("Importance Sample##Importancesampleenvmap", (bool*)&properties.importance_sample_envmap);

		static optix::float3 deltas;
		if (ImmediateGUIDraw::DragFloat3("Deltas##EnvmapDeltas", (float*)&deltas, 1.0f, -180.0f, 180.0f))
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
    const char * prog = ray_type == RAY_TYPE_SHADOW ? "miss_shadow" : "miss";
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
    if (environment_sampler.get() != nullptr)
    {
        Logger::info << "Presampling envmaps... (size " << std::to_string(texture_width) << " " << std::to_string(texture_height) << ")" << std::endl;
        context->launch((camera_1), texture_width, texture_height);
        Logger::info << "Step 1 complete." << std::endl;
        context->launch((camera_2), texture_width, texture_height);
        Logger::info << "Step 2 complete." << std::endl;
        context->launch((camera_3), texture_width, texture_height);
        Logger::info << "Step 3 complete." << std::endl;
        resample_envmaps = false;
        Logger::info << "Done." << std::endl;
    }

}
