//
// Created by alcor on 5/7/18.
//

#include "geometry.h"
#include "folders.h"
#include "optix_utils.h"

Geometry::~Geometry()
{
    mIntersectProgram->destroy();
    mBoundingboxProgram->destroy();
    mBBoxBuffer->destroy();
    mGeometry->destroy();
}

Geometry::Geometry(optix::Context ctx)
{
    mContext = ctx;
    static int id = 0;
    mMeshID = id++;
}

void Geometry::create_and_bind_optix_data()
{
    if (!mIntersectProgram.get())
    {
        std::string path = get_path_ptx("triangle_mesh.cu");
        mIntersectProgram = mContext->createProgramFromPTXFile(path, "mesh_intersect");
    }

    if (!mBoundingboxProgram.get())
    {
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

optix::Geometry Geometry::get_geometry()
{
    if(mGeometry.get() == nullptr)
        create_and_bind_optix_data();
    return mGeometry;
}

void Geometry::load(optix::GeometryInstance & instance)
{
	load_data(instance.get());
}
