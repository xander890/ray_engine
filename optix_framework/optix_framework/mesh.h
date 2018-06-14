#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"
#include <memory>
#include "rendering_method.h"
#include "host_material.h"
#include "transform.h"
#include "optix_serialize.h"
#include "optix_utils.h"
#include "geometry.h"

struct MeshData
{
    optix::Buffer mVbuffer;
    optix::Buffer mNbuffer;
    optix::Buffer mTBuffer;
    optix::Buffer mVIbuffer;
    optix::Buffer mNIbuffer;
    optix::Buffer mTIbuffer;
    int mNumTriangles;
    optix::Aabb   mBoundingBox;
};

namespace cereal
{
template<class Archive>  void save(Archive & ar, MeshData const & m)
{
    throw Exception("Non implemented");
}

template<>
inline void save(cereal::XMLOutputArchiveOptix & ar, MeshData const & m)
{
    save_buffer(ar, m.mVbuffer, "vertex_buffer");
    save_buffer(ar, m.mNbuffer, "normal_buffer");
    save_buffer(ar, m.mTBuffer, "texcoord_buffer");
    save_buffer(ar, m.mVIbuffer, "vertex_index_buffer");
    save_buffer(ar, m.mNIbuffer, "normal_index_buffer");
    save_buffer(ar, m.mTIbuffer, "texture_index_buffer");
    ar(cereal::make_nvp("num_triangles", m.mNumTriangles));
}
}


namespace cereal {

template<class Archive>  void load(Archive & ar, MeshData & m)
{
    throw Exception("Non implemented");
}

template<>
inline void load(cereal::XMLInputArchiveOptix & ar, MeshData & m)
{
    load_buffer(ar, m.mVbuffer, "vertex_buffer");
    load_buffer(ar, m.mNbuffer, "normal_buffer");
    load_buffer(ar, m.mTBuffer, "texcoord_buffer");
    load_buffer(ar, m.mVIbuffer, "vertex_index_buffer");
    load_buffer(ar, m.mNIbuffer, "normal_index_buffer");
    load_buffer(ar, m.mTIbuffer, "texture_index_buffer");
    ar(cereal::make_nvp("num_triangles", m.mNumTriangles));
}
}



class MeshGeometry : public Geometry
{
public:
    MeshGeometry(optix::Context ctx);
    ~MeshGeometry();

    MeshData mMeshData;
    void get_flattened_vertices(std::vector<optix::float3> & triangles) override;
    void init(const char* name, MeshData meshdata);

    virtual void load() override;
    virtual bool on_draw() override;

private:
	void load_data(optix::ScopedObj * obj) override;
    MeshGeometry() {}

	friend class cereal::access;
	// Serialization
    void load( cereal::XMLInputArchiveOptix & archive)
    {
        archive(cereal::virtual_base_class<Geometry>(this));
        archive(cereal::make_nvp("geometry", mMeshData));
        load();
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::virtual_base_class<Geometry>(this));
        archive(cereal::make_nvp("geometry", mMeshData));
    }

};

CEREAL_CLASS_VERSION(MeshGeometry, 0)
CEREAL_REGISTER_TYPE(MeshGeometry)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Geometry, MeshGeometry)
