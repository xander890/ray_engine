#include "chull.h"
#include "folders.h"
#include "optix_utils.h"

void ConvexHull::create_and_bind_optix_data()
{
	if (!mIntersectProgram.get()) {
		std::string path = get_path_ptx("chull.cu");
		mIntersectProgram = mContext->createProgramFromPTXFile(path, "chull_intersect");
	}

	if (!mBoundingboxProgram.get()) {
		std::string path = get_path_ptx("chull.cu");
		mBoundingboxProgram = mContext->createProgramFromPTXFile(path, "chull_bounds");
	}
	if (!mBBoxBuffer.get())
	{
		mBBoxBuffer = create_buffer<optix::Aabb>(mContext);
	}
	if (!mGeometry)
	{
		mGeometry = mContext->createGeometry();
	}

	if(!mPlaneBuffer)
	{
		mPlaneBuffer = mContext->createBuffer(RT_BUFFER_INPUT);
	}

}

bool ConvexHull::on_draw()
{
	return false;
}

ConvexHull::ConvexHull(optix::Context ctx): Geometry(ctx)
{
}

ConvexHull::~ConvexHull()
{
	mPlaneBuffer->destroy();
}

void ConvexHull::load()
{
	if (!mReloadGeometry)
		return;

	create_and_bind_optix_data();

	mPlanes.clear();
	mBoundingBox.invalidate();
	make_planes(mPlanes, mBoundingBox);
	size_t nsides = mPlanes.size();

	mPlaneBuffer->setFormat(RT_FORMAT_FLOAT4);
	mPlaneBuffer->setSize(nsides);

	float4* chplane = (float4*)mPlaneBuffer->map();

	for (int i = 0; i < nsides; i++) {
		float3 p = mPlanes[i].point;
		float3 n = normalize(mPlanes[i].normal);
		chplane[i] = make_float4(n, -dot(n, p));
	}

	mPlaneBuffer->unmap();
	load_data(mGeometry.get());

	mGeometry->setPrimitiveCount(1);
	mGeometry->setIntersectionProgram(mIntersectProgram);
	mGeometry->setBoundingBoxProgram(mBoundingboxProgram);
	initialize_buffer<optix::Aabb>(mBBoxBuffer, mBoundingBox);
	mReloadGeometry = false;
}

void ConvexHull::load_data(optix::ScopedObj * obj)
{
	get_var(obj, "planes")->setBuffer(mPlaneBuffer);
	get_var(obj, "chull_bbmin")->setFloat(mBoundingBox.m_min);
	get_var(obj, "chull_bbmax")->setFloat(mBoundingBox.m_max);
}

void ConvexHull::load(optix::GeometryInstance & instance)
{
}

