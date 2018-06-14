#include "sphere.h"
#include "folders.h"
#include "optix_utils.h"
#include "immediate_gui.h"

void Sphere::load_data(optix::ScopedObj * obj)
{
	get_var(obj, "center")->setFloat(center);
	get_var(obj, "radius")->setFloat(radius);
}

void Sphere::create_and_bind_optix_data()
{
	if (!mIntersectProgram.get()) {
		std::string path = get_path_ptx("sphere.cu");
		mIntersectProgram = mContext->createProgramFromPTXFile(path, "sphere_intersect");
	}

	if (!mBoundingboxProgram.get()) {
		std::string path = get_path_ptx("sphere.cu");
		mBoundingboxProgram = mContext->createProgramFromPTXFile(path, "sphere_bounds");
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

bool Sphere::on_draw()
{
	if(ImmediateGUIDraw::InputFloat3("Center", &center.x))
	{
		mReloadGeometry = true;
	}

	if(ImmediateGUIDraw::InputFloat("Radius", &radius))
	{
		mReloadGeometry = true;
	}

	return mReloadGeometry;
}

void Sphere::load()
{
	if (!mReloadGeometry)
		return;

	create_and_bind_optix_data();

	mBoundingBox.include(center - make_float3(radius));
	mBoundingBox.include(center + make_float3(radius));

	mGeometry->setPrimitiveCount(1);
	mGeometry->setIntersectionProgram(mIntersectProgram);
	mGeometry->setBoundingBoxProgram(mBoundingboxProgram);
	mGeometry->markDirty();
	initialize_buffer<optix::Aabb>(mBBoxBuffer, mBoundingBox);
	mReloadGeometry = false;

	load_data(mGeometry.get());
}

