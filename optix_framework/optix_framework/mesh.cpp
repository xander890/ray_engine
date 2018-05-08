#include "mesh.h"
#include "shader_factory.h"
#include "material_library.h"
#include "scattering_material.h"
#include <algorithm>
#include<host_material.h>
#include "optix_utils.h"
#include "immediate_gui.h"
#include <array>
#include "object_host.h"

MeshGeometry::MeshGeometry(optix::Context ctx) : Geometry(ctx)
{

}

void MeshGeometry::init(const char* name, MeshData meshdata)
{
    mMeshName = name;
    mMeshData = meshdata;
    // Load triangle_mesh programs
	load();
}

void MeshGeometry::load()
{
	if (!mReloadGeometry)
		return;
    create_and_bind_optix_data();

    mGeometry->setPrimitiveCount(mMeshData.mNumTriangles);
    mGeometry->setIntersectionProgram(mIntersectProgram);
    mGeometry->setBoundingBoxProgram(mBoundingboxProgram);
    mGeometry["vertex_buffer"]->setBuffer(mMeshData.mVbuffer);
    mGeometry["vindex_buffer"]->setBuffer(mMeshData.mVIbuffer);
    mGeometry["normal_buffer"]->setBuffer(mMeshData.mNbuffer);
    mGeometry["nindex_buffer"]->setBuffer(mMeshData.mNIbuffer);
    mGeometry["texcoord_buffer"]->setBuffer(mMeshData.mTBuffer);
    mGeometry["tindex_buffer"]->setBuffer(mMeshData.mTIbuffer);
    mGeometry["num_triangles"]->setUint(mMeshData.mNumTriangles);
    mGeometry->markDirty();
    initialize_buffer<optix::Aabb>(mBBoxBuffer, mMeshData.mBoundingBox);
	mReloadGeometry = false;
}

bool MeshGeometry::on_draw()
{
	bool changed = false;
    ImmediateGUIDraw::TextWrapped("Triangles: %d\n", mMeshData.mNumTriangles);
	return changed;
}

void MeshGeometry::get_flattened_vertices(std::vector<optix::float3> &triangles)
{
    optix::float3 * vertices = reinterpret_cast<optix::float3*>(mMeshData.mVbuffer->map());
    optix::int3 * vertices_indices = reinterpret_cast<optix::int3*>(mMeshData.mVIbuffer->map());

    for(int i = 0; i < mMeshData.mNumTriangles; i++)
    {
        optix::int3 vi = vertices_indices[i];
        triangles.push_back(vertices[vi.x]);
        triangles.push_back(vertices[vi.y]);
        triangles.push_back(vertices[vi.z]);
    }

    mMeshData.mVbuffer->unmap();
    mMeshData.mVIbuffer->unmap();
}

MeshGeometry::~MeshGeometry()
{
    // FIXME
    //mMeshData.mVbuffer->destroy();
    mMeshData.mVIbuffer->destroy();
    //mMeshData.mNbuffer->destroy();
    mMeshData.mNIbuffer->destroy();
    //mMeshData.mTBuffer->destroy();
    mMeshData.mTIbuffer->destroy();
}
