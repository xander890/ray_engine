#include <device_common_data.h>
#include<material.h>

rtDeclareVariable(optix::float3, ior_complex_real_sq, , );
rtDeclareVariable(optix::float3, ior_complex_imag_sq, , );
rtDeclareVariable(MaterialDataCommon, main_material, , );

rtBuffer<MaterialDataCommon, 1> material_buffer;

__device__ __forceinline__ const MaterialDataCommon& get_material()
{
    float3 hit_pos = ray.origin + t_hit * ray.direction;
    int material_idx = (hit_pos.z > 0) ? 0 : 1;
    material_idx = optix::min(material_idx, (int)material_buffer.size() - 1);
    return material_buffer[material_idx];
}

__device__ __forceinline__ const MaterialDataCommon& get_material(const float3 & hit_pos)
{
    int material_idx = (hit_pos.z > 0) ? 0 : 1;
    material_idx = optix::min(material_idx, (int)material_buffer.size() - 1);
    return material_buffer[material_idx];
}