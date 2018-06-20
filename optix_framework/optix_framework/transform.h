#pragma once
#include <memory>
#include <optix_world.h>
#include "optix_serialize_utils.h"

class Transform
{
public:
	Transform(optix::Context & ctx);
	~Transform();

	optix::Matrix4x4 get_matrix();
	bool on_draw();
	bool has_changed() const;
	void translate(const optix::float3 & t);
	void rotate(float angle, const optix::float3 & axis);
	void scale(const optix::float3 & s);
	void load();
	optix::Transform get_transform() { return mTransform; }

private:
	optix::float3 mTranslation = optix::make_float3(0,0,0);
	optix::float3 mScale = optix::make_float3(1);
	optix::float3 mRotationAxis = optix::make_float3(0, 0, 1);
	float mRotationAngle = 0;
	int id;
	bool mHasChanged = true;

	friend class cereal::access;

    static void load_and_construct( cereal::XMLInputArchiveOptix & archive, cereal::construct<Transform> & construct )
    {
        construct(archive.get_context());
        archive(
                cereal::make_nvp("translation", construct->mTranslation),
                cereal::make_nvp("scale", construct->mScale),
                cereal::make_nvp("rotation_axis", construct->mRotationAxis),
                cereal::make_nvp("rotation_angle", construct->mRotationAngle)
        );
        construct->load();
    }

	template<class Archive>
	void save(Archive & archive) const
	{
		archive(
                cereal::make_nvp("translation", mTranslation),
                cereal::make_nvp("scale", mScale),
                cereal::make_nvp("rotation_axis", mRotationAxis),
                cereal::make_nvp("rotation_angle", mRotationAngle)
        );
	}
	optix::Context context;
	optix::Transform mTransform = nullptr;
};

