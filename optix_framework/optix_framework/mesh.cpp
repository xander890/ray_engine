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

Geometry::Geometry(optix::Context ctx) : mContext(ctx)
{
	static int id = 0;
	mMeshID = id++;
}

void Geometry::init(const char* name, MeshData meshdata)
{
    mMeshName = name;
    mMeshData = meshdata;
    // Load triangle_mesh programs
	load();
}

void Geometry::load()
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



void Geometry::create_and_bind_optix_data()
{
    if (!mIntersectProgram.get()) {
        std::string path = get_path_ptx("triangle_mesh.cu");
        mIntersectProgram = mContext->createProgramFromPTXFile(path, "mesh_intersect");
    }

    if (!mBoundingboxProgram.get()) {
        std::string path = get_path_ptx("triangle_mesh.cu");
        mBoundingboxProgram = mContext->createProgramFromPTXFile(path, "mesh_bounds");
    }
    if (!mBBoxBuffer.get())
    {
        mBBoxBuffer = create_buffer<optix::Aabb>(mContext);
    }
    if (!mGeometry)
    {
        mGeometry = mContext->createGeometry();
    }
}


bool Geometry::on_draw()
{
	bool changed = false;
	return changed;
}

void Geometry::get_flattened_vertices(std::vector<optix::float3> &triangles)
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

Geometry::~Geometry()
{
    mIntersectProgram->destroy();
    mBoundingboxProgram->destroy();
    mBBoxBuffer->destroy();
    // FIXME
    //mMeshData.mVbuffer->destroy();
    mMeshData.mVIbuffer->destroy();
    //mMeshData.mNbuffer->destroy();
    mMeshData.mNIbuffer->destroy();
    //mMeshData.mTBuffer->destroy();
    mMeshData.mTIbuffer->destroy();
    mGeometry->destroy();
}
