#pragma once

#include "device_common_data.h"

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   vindex_buffer;    // position indices 
rtBuffer<int3>   nindex_buffer;    // normal indices
rtBuffer<int3>   tindex_buffer;    // texcoord indices
rtDeclareVariable(unsigned int, num_triangles, , );
rtDeclareVariable(BufPtr<optix::Aabb>, local_bounding_box, , );
rtDeclareVariable(rtObject, current_geometry_node, , );
rtDeclareVariable(int, mesh_id, , );