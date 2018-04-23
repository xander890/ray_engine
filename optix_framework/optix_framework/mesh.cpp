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
	load();
}

void Geometry::load_geometry()
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
	mReloadGeometry = false;
}

void Geometry::load()
{
	load_geometry();
}




void Geometry::create_and_bind_optix_data()
{
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
