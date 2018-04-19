#pragma once
#include <memory>
#include <optix_world.h>
#include "optix_serialize.h"

class Transform : public std::enable_shared_from_this<Transform>
{
public:
	Transform();
	~Transform();
	Transform(Transform&&) noexcept;
	Transform& operator=(Transform&&) noexcept;

	optix::Matrix4x4 get_matrix();
	bool on_draw();
	bool has_changed() const;
	void translate(const optix::float3 & t);
	void rotate(float angle, const optix::float3 & axis);
	void scale(const optix::float3 & s);

private:
	optix::float3 mTranslation = optix::make_float3(0,0,0);
	optix::float3 mScale = optix::make_float3(1);
	optix::float3 mRotationAxis = optix::make_float3(0, 0, 1);
	float mRotationAngle = 0;
	int id;
	bool mHasChanged = true;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(CEREAL_NVP(mTranslation), CEREAL_NVP(mScale), CEREAL_NVP(mRotationAngle), CEREAL_NVP(mRotationAngle));
	}
};

