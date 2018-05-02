#pragma once
#include <optix_world.h>
#include "shader.h"
#include "enums.h"
#include <memory>
#include "rendering_method.h"
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include "cereal/types/memory.hpp"
#include "host_material.h"
#include "transform.h"
#include "optix_serialize.h"
#include "optix_utils.h"

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

inline void save_buffer(cereal::XMLOutputArchive & archive, optix::Buffer buffer, std::string name)
{
    void * data = buffer->map();
    RTsize dim = buffer->getDimensionality();
    std::vector<RTsize> dims = std::vector<RTsize>(dim);
    buffer->getSize(dim, &dims[0]);
    RTsize total_size = 1;
    for(int i = 0; i < dim; i++)
        total_size *= dims[i];

    RTsize element = buffer->getElementSize();
    archive.saveBinaryValue(data, total_size * element, name.c_str());
    buffer->unmap();
    archive(cereal::make_nvp(name + "_element_size", element));
    archive(cereal::make_nvp(name + "_dimensionality", dim));
    archive(cereal::make_nvp(name + "_size", dims));
}

namespace cereal
{
template<class Archive>  void save(Archive & ar, MeshData const & m)
{
    throw Exception("Non implemented");
}

template<>
inline void save(cereal::XMLOutputArchive & ar, MeshData const & m)
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

inline void load_buffer(cereal::XMLInputArchiveOptix & archive, optix::Buffer & buffer, std::string name)
{
    RTsize element, dim;
    archive(cereal::make_nvp(name + "_element_size", element));
    archive(cereal::make_nvp(name + "_dimensionality", dim));

    std::vector<RTsize> dims = std::vector<RTsize>(dim);
    archive(cereal::make_nvp(name + "_size", dims));

    buffer = archive.get_context()->createBuffer(RT_BUFFER_INPUT);
    buffer->setFormat(RT_FORMAT_USER);
    buffer->setSize(dim, &dims[0]);
    buffer->setElementSize(element);

    RTsize total_size = 1;
    for(int i = 0; i < dim; i++)
        total_size *= dims[i];

    void * data = buffer->map();
    archive.loadBinaryValue(data, total_size * element, name.c_str());
    buffer->unmap();
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

class Geometry
{
public:
    explicit Geometry(optix::Context ctx);
    ~Geometry();

    void init(const char* name, MeshData meshdata);

    optix::Geometry mGeometry = nullptr;
    optix::Context  mContext;

	void load();
	bool on_draw();
    MeshData mMeshData;
    optix::Geometry get_geometry() { return mGeometry; }

    void get_flattened_vertices(std::vector<optix::float3> & triangles);
    optix::Buffer get_bounding_box_buffer() {return mBBoxBuffer; }

private:

    void create_and_bind_optix_data();
    optix::Program         mIntersectProgram;
    optix::Program         mBoundingboxProgram;
    optix::Buffer          mBBoxBuffer;
    std::string            mMeshName;

	bool mReloadGeometry = true;

	friend class cereal::access;
	// Serialization
    static void load_and_construct( cereal::XMLInputArchiveOptix & archive, cereal::construct<Geometry> & construct )
    {
        construct(archive.get_context());
        archive(cereal::make_nvp("name", construct->mMeshName));
        archive(cereal::make_nvp("geometry", construct->mMeshData));
        construct->load();
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("name", mMeshName));
        archive(cereal::make_nvp("geometry", mMeshData));
    }

    int mMeshID;
};
