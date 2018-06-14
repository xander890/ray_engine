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
	load_data(mGeometry.get());
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

void MeshGeometry::load_data(optix::ScopedObj * obj)
{
	get_var(obj, "vertex_buffer")->setBuffer(mMeshData.mVbuffer);
	get_var(obj, "vindex_buffer")->setBuffer(mMeshData.mVIbuffer);
	get_var(obj, "normal_buffer")->setBuffer(mMeshData.mNbuffer);
	get_var(obj, "nindex_buffer")->setBuffer(mMeshData.mNIbuffer);
	get_var(obj, "texcoord_buffer")->setBuffer(mMeshData.mTBuffer);
	get_var(obj, "tindex_buffer")->setBuffer(mMeshData.mTIbuffer);
	get_var(obj, "num_triangles")->setUint(mMeshData.mNumTriangles);
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
