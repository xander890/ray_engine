#include "transform.h"
#include "immediate_gui.h"

struct Transform::Implementation
{
	optix::float3 mTranslation = optix::make_float3(0,0,0);
	optix::float3 mScale = optix::make_float3(1);
	optix::float3 mRotationAxis = optix::make_float3(0, 0, 1);
	float mRotationAngle = 0;
	int id;
	bool mHasChanged = true;
};


Transform::Transform() {
	static int id = 0;
	impl = std::make_unique<Implementation>();
	impl->id = id++;
}

Transform::~Transform() = default;
Transform::Transform(Transform&&) noexcept = default;
Transform& Transform::operator=(Transform&&) noexcept = default;

optix::Matrix4x4 Transform::get_matrix()
{
	impl->mHasChanged = false;
	auto mtx = optix::Matrix4x4::identity();
	mtx = mtx.scale(impl->mScale);
	mtx = mtx.rotate(impl->mRotationAngle, impl->mRotationAxis);
	mtx = mtx.translate(impl->mTranslation);
	return mtx;
}

bool Transform::on_draw()
{
	impl->mHasChanged |= ImmediateGUIDraw::InputFloat3("Translate##TranslateTransform" + impl->id, &impl->mTranslation.x , 2);

	static optix::float3 val = impl->mRotationAxis;
	if (ImmediateGUIDraw::InputFloat3("Rotation axis##RotationAxisTransform" + impl->id, &val.x, 2))
	{
		impl->mHasChanged = true;
		impl->mRotationAxis = optix::normalize(val);
	}

	impl->mHasChanged |= ImmediateGUIDraw::InputFloat("Rotation angle##RotationAngleTransform" + impl->id, &impl->mRotationAngle, 2);
	impl->mHasChanged |= ImmediateGUIDraw::InputFloat3("Scale##ScaleTransform" + impl->id, &impl->mScale.x, 2);
	return impl->mHasChanged;
}

bool Transform::has_changed() const
{
	return impl->mHasChanged;
}

void Transform::translate(const optix::float3 & t)
{
	impl->mHasChanged = true;
	impl->mTranslation = t;
}

void Transform::rotate(float angle, const optix::float3 & axis)
{
	impl->mHasChanged = true;
	impl->mRotationAngle = angle;
	impl->mRotationAxis = axis;
}

void Transform::scale(const optix::float3 & s)
{
	impl->mHasChanged = true;
	impl->mScale = s;
}
