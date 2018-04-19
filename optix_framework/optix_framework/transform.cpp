#include "transform.h"
#include "immediate_gui.h"

Transform::Transform() {
}

Transform::~Transform() = default;
Transform::Transform(Transform&&) noexcept = default;
Transform& Transform::operator=(Transform&&) noexcept = default;

optix::Matrix4x4 Transform::get_matrix()
{
	mHasChanged = false;
	auto mtx = optix::Matrix4x4::identity();
	mtx = mtx.scale(mScale);
	mtx = mtx.rotate(mRotationAngle, mRotationAxis);
	mtx = mtx.translate(mTranslation);
	return mtx;
}

bool Transform::on_draw()
{
	mHasChanged |= ImmediateGUIDraw::InputFloat3("Translate##TranslateTransform" + id, &mTranslation.x , 2);

	static optix::float3 val = mRotationAxis;
	if (ImmediateGUIDraw::InputFloat3("Rotation axis##RotationAxisTransform" + id, &val.x, 2))
	{
		mHasChanged = true;
		mRotationAxis = optix::normalize(val);
	}

	mHasChanged |= ImmediateGUIDraw::InputFloat("Rotation angle##RotationAngleTransform" + id, &mRotationAngle, 2);
	mHasChanged |= ImmediateGUIDraw::InputFloat3("Scale##ScaleTransform" + id, &mScale.x, 2);
	return mHasChanged;
}

bool Transform::has_changed() const
{
	return mHasChanged;
}

void Transform::translate(const optix::float3 & t)
{
	mHasChanged = true;
	mTranslation = t;
}

void Transform::rotate(float angle, const optix::float3 & axis)
{
	mHasChanged = true;
	mRotationAngle = angle;
	mRotationAxis = axis;
}

void Transform::scale(const optix::float3 & s)
{
	mHasChanged = true;
	mScale = s;
}
