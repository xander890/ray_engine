#pragma once
#include <memory>
#include <optix_world.h>

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
	struct Implementation;
	std::unique_ptr<Implementation> impl;
};

