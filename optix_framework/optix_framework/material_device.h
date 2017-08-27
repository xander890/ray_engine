#include <device_common_data.h>
#include<material.h>

rtDeclareVariable(optix::float3, ior_complex_real_sq, , );
rtDeclareVariable(optix::float3, ior_complex_imag_sq, , );
rtDeclareVariable(MaterialDataCommon, main_material, , );

rtBuffer<MaterialDataCommon, 1> material_buffer;

rtDeclareVariable(int, noise_tex, , );
rtDeclareVariable(float, noise_scale, , );

__device__ __forceinline__ const MaterialDataCommon& get_material()
{
    float3 hit_pos = ray.origin + t_hit * ray.direction;
    int material_idx = rtTex3D<float4>(noise_tex, hit_pos.x * noise_scale, hit_pos.y* noise_scale, hit_pos.z*noise_scale).x > 0.5 ? 0 : 1;
    material_idx = optix::min(material_idx, (int)material_buffer.size() - 1);
    return material_buffer[material_idx];
}

__device__ __forceinline__ const MaterialDataCommon& get_material(const float3 & hit_pos)
{
    int material_idx = rtTex3D<float4>(noise_tex, hit_pos.x * noise_scale, hit_pos.y* noise_scale, hit_pos.z*noise_scale).x > 0.5 ? 0 : 1;
    material_idx = optix::min(material_idx, (int)material_buffer.size() - 1);
    return material_buffer[material_idx];
}