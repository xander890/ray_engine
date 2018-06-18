#pragma once
#include <optixu/optixu_aabb.h>
#include <optixu/optixpp_namespace.h>
#include <vector>
#include "geometry.h"

/*
 * Utility struct for a convex hull, representing a plane. class just owns a point and a normal, plus some serialization capabilities.
 */
struct Plane
{
	float3 point;
	float3 normal;

private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar)
    {
        ar(cereal::make_nvp("origin", point), cereal::make_nvp("normal", normal));
    }
};

/*
 * Abstract Geometry class representing a procedural convex hull. A convex hull, in this case, represents a volume of space enclosed in between a number planes. The number of planes can be variable, and each implementing class can reimplement the abstract function make_planes to define its own planes.
 */
class ConvexHull : public Geometry
{
public:
	ConvexHull(optix::Context ctx);
	virtual ~ConvexHull();

	void load() override;
	void load(optix::GeometryInstance & instance) override;
    bool on_draw() override;

protected:
	void load_data(optix::ScopedObj * obj) override;
    void create_and_bind_optix_data() override;

    optix::Aabb mBoundingBox;
    optix::Buffer mPlaneBuffer;
    std::vector<Plane> mPlanes;

	virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bbox) = 0;
    ConvexHull() : Geometry() {}

private:
	// Serialization methods, just pass it through to the geometry class.
	friend class cereal::access;
    void load( cereal::XMLInputArchiveOptix & archive)
    {
        archive(cereal::virtual_base_class<Geometry>(this));
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::virtual_base_class<Geometry>(this));
    }

};

CEREAL_CLASS_VERSION(ConvexHull, 0)
CEREAL_REGISTER_TYPE(ConvexHull)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Geometry, ConvexHull)
