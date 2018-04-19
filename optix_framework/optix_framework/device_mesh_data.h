#pragma once

#include "device_common_data.h"

rtBuffer<optix::float3> vertex_buffer;
rtBuffer<optix::float3> normal_buffer;
rtBuffer<optix::float2> texcoord_buffer;
rtBuffer<optix::int3>   vindex_buffer;    // position indices
rtBuffer<optix::int3>   nindex_buffer;    // normal indices
rtBuffer<optix::int3>   tindex_buffer;    // texcoord indices
rtDeclareVariable(unsigned int, num_triangles, , );
rtDeclareVariable(BufPtr<optix::Aabb>, local_bounding_box, , );
rtDeclareVariable(rtObject, current_geometry_node, , );
rtDeclareVariable(int, mesh_id, , );