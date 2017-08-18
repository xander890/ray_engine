#include "environment_map_background.h"

#include "folders.h"
#include "host_device_common.h"
#include "environment_map.h"
#include <cmath>
#include "enums.h"
#include "SampleScene.h"
#include "CGLA/CGLA.h"
#include "CGLA/Mat3x3f.h"
#include "ImageLoader.h"

using namespace optix;
void EnvironmentMap::init(optix::Context & ctx)
{
    MissProgram::init(ctx);
    context = ctx;
    envmap_deltas = ParameterParser::get_parameter<float3>("light", "envmap_deltas", make_float3(0), "Rotation offsetof environment map.");
    properties.lightmap_multiplier = ParameterParser::get_parameter<float3>("light", "lightmap_multiplier", make_float3(1.0f), "Environment map multiplier");
    bool is_env = false;
    is_env = true;
    std::string ptx_path = get_path_ptx("env_cameras.cu");
    environment_sampler = loadTexture(context->getContext(), Folders::texture_folder + envmap_file, make_float3(1.0f));
    properties.environment_map_tex_id = environment_sampler->getId();
    properties.importance_sample_envmap = 1;

    RTsize w, h;
    environment_sampler.get()->getBuffer(0, 0)->getSize(w, h);
    texture_width = w;
    texture_height = h;

    sampling_properties.env_luminance = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_f = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.marginal_pdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_pdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    sampling_properties.marginal_cdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_height)->getId());
    sampling_properties.conditional_cdf = (context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, texture_width, texture_height)->getId());
    
    Program ray_gen_program_1 = ctx->createProgramFromPTXFile(ptx_path, "env_luminance_camera");
    ctx->setRayGenerationProgram(as_integer(CameraType::ENV_1), ray_gen_program_1);   
    Program ray_gen_program_2 = ctx->createProgramFromPTXFile(ptx_path, "env_marginal_camera");
    ctx->setRayGenerationProgram(as_integer(CameraType::ENV_2), ray_gen_program_2);
    Program ray_gen_program_3 = ctx->createProgramFromPTXFile(ptx_path, "env_pdf_camera");
    ctx->setRayGenerationProgram(as_integer(CameraType::ENV_3), ray_gen_program_3);    

    property_buffer = ctx->createBuffer(RT_BUFFER_INPUT);
    property_buffer->setFormat(RT_FORMAT_USER);
    property_buffer->setElementSize(sizeof(EnvmapProperties));
    property_buffer->setSize(1);
    memcpy(property_buffer->map(), &properties, sizeof(EnvmapProperties));
    property_buffer->unmap();

    BufPtr<EnvmapProperties> b = BufPtr<EnvmapProperties>(property_buffer->getId());
    ctx["envmap_properties"]->setUserData(sizeof(BufPtr<EnvmapProperties>), &b);

    sampling_property_buffer = ctx->createBuffer(RT_BUFFER_INPUT);
    sampling_property_buffer->setFormat(RT_FORMAT_USER);
    sampling_property_buffer->setElementSize(sizeof(EnvmapProperties));
    sampling_property_buffer->setSize(1);
    memcpy(sampling_property_buffer->map(), &sampling_properties, sizeof(EnvmapProperties));
    sampling_property_buffer->unmap();

    BufPtr<EnvmapProperties> b2 = BufPtr<EnvmapProperties>(sampling_property_buffer->getId());
    ctx["envmap_importance_sampling"]->setUserData(sizeof(BufPtr<EnvmapProperties>), &b2);
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

void EnvironmentMap::set_into_gui(GUI * gui)
{
    const char* env_map_correction_group = "Environment map";
    gui->addFloatVariableCallBack("Lightmap multiplier", setLightMultiplier ,getLightMultiplier, this, env_map_correction_group);
    gui->addCheckBox("Importance sample", (bool*)&properties.importance_sample_envmap, env_map_correction_group);
    gui->addFloatVariableCallBack("Delta X", setDeltaX, getDeltaX, this, env_map_correction_group, -180.0, 180.0f, .010f);
    gui->addFloatVariableCallBack("Delta Y", setDeltaY, getDeltaY, this, env_map_correction_group, -180.0, 180.0f, .010f);
    gui->addFloatVariableCallBack("Delta Z", setDeltaZ, getDeltaZ, this, env_map_correction_group, -180.0, 180.0f, .010f);
}

void EnvironmentMap::remove_from_gui(GUI* gui)
{
}

bool EnvironmentMap::get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program)
{
    const char * prog = ray_type == RAY_TYPE_SHADOW ? "miss_shadow" : "miss";
    program = ctx->createProgramFromPTXFile(get_path_ptx("environment_map_background.cu"), prog);
    return true;
}


void EnvironmentMap::setDeltaX(const void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    scene->envmap_deltas.x = *(float*)var / 180.0f * M_PIf;
    scene->resample_envmaps = true;
  // FIXME  scene->reset_renderer();
}

void EnvironmentMap::setDeltaY(const void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    scene->envmap_deltas.y = *(float*)var / 180.0f * M_PIf;
    scene->resample_envmaps = true;
    // FIXME    scene->reset_renderer();
}


void EnvironmentMap::getDeltaX(void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    *(float*)var = scene->envmap_deltas.x * 180.0f / M_PIf;
}

void EnvironmentMap::getDeltaY(void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    *(float*)var = scene->envmap_deltas.y * 180.0f / M_PIf;
}

void EnvironmentMap::setDeltaZ(const void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    scene->envmap_deltas.z = *(float*)var / 180.0f * M_PIf;
    scene->resample_envmaps = true;

    // FIXME    scene->reset_renderer();
}

void EnvironmentMap::getDeltaZ(void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    *(float*)var = scene->envmap_deltas.z * 180.0f / M_PIf;
}

void EnvironmentMap::setLightMultiplier(const void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    float v = *((float*)var);
    scene->properties.lightmap_multiplier = make_float3(v);
}

void EnvironmentMap::getLightMultiplier(void* var, void* data)
{
    EnvironmentMap* scene = reinterpret_cast<EnvironmentMap*>(data);
    *(float*)var = scene->properties.lightmap_multiplier.x;
}

Matrix3x3 get_offset_lightmap_rotation_matrix(float delta_x, float delta_y, float delta_z, const optix::Matrix3x3& current_matrix)
{
    CGLA::Mat3x3f matrix = CGLA::rotation_Mat3x3f(CGLA::ZAXIS, delta_z) * CGLA::rotation_Mat3x3f(CGLA::YAXIS, delta_y) * rotation_Mat3x3f(CGLA::XAXIS, delta_x);
    Matrix3x3 optix_matrix = *reinterpret_cast<optix::Matrix3x3*>(&matrix);
    optix_matrix = optix_matrix * current_matrix;
    return optix_matrix;
}

void EnvironmentMap::presample_environment_map()
{
    properties.lightmap_rotation_matrix = get_offset_lightmap_rotation_matrix(envmap_deltas.x, envmap_deltas.y, envmap_deltas.z, properties.lightmap_rotation_matrix);
    // Environment importance sampling pre-pass
    if (environment_sampler.get() != nullptr)
    {
        Logger::info << "Presampling envmaps... (size " << to_string(texture_width) << " " << to_string(texture_height) << ")" << endl;
        context->launch(as_integer(CameraType::ENV_1), texture_width, texture_height);
        Logger::info << "Step 1 complete." << endl;
        context->launch(as_integer(CameraType::ENV_2), texture_width, texture_height);
        Logger::info << "Step 2 complete." << endl;
        context->launch(as_integer(CameraType::ENV_3), texture_width, texture_height);
        Logger::info << "Step 3 complete." << endl;
        resample_envmaps = false;
        Logger::info << "Done." << endl;
    }

}
