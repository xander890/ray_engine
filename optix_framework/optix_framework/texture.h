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
    void set_size(size_t width, size_t height = 0, size_t depth = 0);
    void set_size(size_t dimensions, size_t * dims);

    size_t get_width() const { return mDimensions[0]; }
    size_t get_height() const { return mDimensions[1]; }
    size_t get_depth() const { return mDimensions[2]; }

    TexPtr get_id();
    void set_data(float * data, size_t size);
    optix::float4 get_texel(size_t x, size_t y = 0, size_t z = 0) const;

private:
    optix::Buffer textureBuffer;
    optix::TextureSampler textureSampler;
    RTsize mDimensions[3];
    RTsize mDimensionality;
    float * mData = nullptr;
    size_t get_number_of_elements() const { return mDimensions[0]*mDimensions[1]*mDimensions[2]*4; }

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
        size_t dimensions;
        size_t dims[3];
        archive(
                cereal::make_nvp("width", dims[0]),
                cereal::make_nvp("height", dims[1]),
                cereal::make_nvp("depth", dims[2]),
                cereal::make_nvp("dimensionality", dimensions)
        );
        construct->set_size(dimensions, dims);
        float * vals = new float[construct->get_number_of_elements()];
        archive.loadBinaryValue(vals,construct->get_number_of_elements()*sizeof(float), "texture");
        construct->set_data(vals, construct->get_number_of_elements()*sizeof(float));
        delete[] vals;
    }
};

template<>
inline void Texture::save(cereal::XMLOutputArchiveOptix & archive) const
{
    archive(
            cereal::make_nvp("width", mDimensions[0]),
            cereal::make_nvp("height", mDimensions[1]),
            cereal::make_nvp("depth", mDimensions[2]),
            cereal::make_nvp("dimensionality", mDimensionality));
    archive.saveBinaryValue(mData, get_number_of_elements()*sizeof(float), "texture");
}



