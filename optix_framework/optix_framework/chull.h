
#ifndef chull_h__
#define chull_h__

#include <optixu/optixu_aabb.h>
#include <optixu/optixpp_namespace.h>
#include <vector>
#include "geometry.h"

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



class ConvexHull : public Geometry
{
public:
    ConvexHull(optix::Context ctx) : Geometry(ctx)
    {
    }

    virtual ~ConvexHull()
    {
        mPlaneBuffer->destroy();
    }

    void load() override;
    bool on_draw() override;

protected:
    void create_and_bind_optix_data() override;

    optix::Aabb mBoundingBox;
    optix::Buffer mPlaneBuffer;
    std::vector<Plane> mPlanes;


    virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bbox) = 0;

    ConvexHull() : Geometry() {}

private:
    friend class cereal::access;
    // Serialization
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

CEREAL_REGISTER_TYPE(ConvexHull)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Geometry, ConvexHull)


#endif // chull_h__
