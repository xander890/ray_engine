//
// Created by alcor on 4/24/18.
//
#pragma once
#include "host_device_common.h"
#include <cereal/cereal.hpp>
#include "xml_archive.hpp"

class Texture
{
public:
    Texture(optix::Context context);
    ~Texture();
    void set_size(int width, int height = -1, int depth = -1);
    void set_size(int dimensions, int * dims);

    int get_width() const { return width; }
    int get_height() const { return height; }
    int get_depth() const { return depth; }

    TexPtr get_id();
    float * map_data();
    void unmap_data();
    void set_data(float * data, size_t size);

private:
    optix::Buffer textureBuffer;
    optix::TextureSampler textureSampler;
    unsigned int width, height, depth;
    unsigned int dimensions;

    friend class cereal::access;

    template<class Archive>
    void save(Archive & archive) const
    {
        throw std::logic_error("Unsupported archive.");
    }

    template<class Archive>
    static void load_and_construct(Archive & archive, cereal::construct<Texture> & construct)
    {
        throw std::logic_error("Unsupported archive.");
    }

    static void load_and_construct( cereal::XMLInputArchiveOptix & archive, cereal::construct<Texture> & construct )
    {
        construct(archive.get_context());
        int width,height,depth,dimensions;
        archive( CEREAL_NVP(width),  CEREAL_NVP(height),  CEREAL_NVP(depth),  CEREAL_NVP(dimensions));
        int dims[3] = {width, height, depth};
        construct->set_size(dimensions, dims);
        void * vals = construct->textureBuffer->map();
        archive.loadBinaryValue(vals, width*height*depth*4*sizeof(float), "texture");
        construct->textureBuffer->unmap();
    }
};

template<>
inline void Texture::save(cereal::XMLOutputArchiveOptix & archive) const
{
    archive( CEREAL_NVP(width),  CEREAL_NVP(height),  CEREAL_NVP(depth),  CEREAL_NVP(dimensions));
    const void * v = textureBuffer->map();
    archive.saveBinaryValue(v, width*height*depth*4*sizeof(float), "texture");
    textureBuffer->unmap();
}



