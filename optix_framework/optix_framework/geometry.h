#pragma once
#include "host_device_common.h"
#include "optix_serialize.h"

class Geometry
{
public:
    Geometry(optix::Context ctx);
    virtual ~Geometry();
    optix::Geometry get_geometry();
    optix::Buffer get_bounding_box_buffer() {return mBBoxBuffer; }
    virtual void load() = 0;
	virtual void load(optix::GeometryInstance & instance);

    virtual bool on_draw() = 0;

    virtual void get_flattened_vertices(std::vector<optix::float3> & triangles) { triangles.clear(); }

protected:
	virtual void load_data(optix::ScopedObj * obj) = 0;

    optix::Geometry mGeometry = nullptr;
    optix::Context  mContext;
    optix::Program         mIntersectProgram;
    optix::Program         mBoundingboxProgram;
    optix::Buffer          mBBoxBuffer;
    std::string            mMeshName;
    Geometry() {}

    virtual void create_and_bind_optix_data();
    int mMeshID;
    bool mReloadGeometry = true;

private:
    friend class cereal::access;
    // Serialization
    void load( cereal::XMLInputArchiveOptix & archive)
    {
        mContext = archive.get_context();
        archive(cereal::make_nvp("name", mMeshName));
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("name", mMeshName));
    }

};