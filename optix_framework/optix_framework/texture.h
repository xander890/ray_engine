//
// Created by alcor on 4/24/18.
//
#pragma once
#include "host_device_common.h"
#include "optix_serialize.h"

class Texture
{
public:
    enum Format
    {
        BYTE,
        UNSIGNED_BYTE,
        SHORT,
        UNSIGNED_SHORT,
        INT,
        UNSIGNED_INT,
        FLOAT
    };

    Texture(optix::Context context, const Texture::Format & format = FLOAT, const RTsize & element_size = 4);
    ~Texture();
    void set_size(size_t width, size_t height = 0, size_t depth = 0);
    void set_size(size_t dimensions, size_t * dims);

    void set_format(const RTformat& format);
    void set_format(RTsize elements, const Texture::Format & format);

    size_t get_width() const { return mDimensions[0]; }
    size_t get_height() const { return mDimensions[1]; }
    size_t get_depth() const { return mDimensions[2]; }

    TexPtr get_id();
    void set_data(void * data, size_t size);

    template<typename T>
    T get_texel(size_t x, size_t y = 0, size_t z = 0) const;
    template<typename T>
    void set_texel(const T& val, size_t x, size_t y = 0, size_t z = 0);

    optix::TextureSampler get_sampler() { return textureSampler; }

    void update();

private:
    int mID;
    optix::Buffer textureBuffer;
    optix::TextureSampler textureSampler;
    RTsize mDimensions[3] = {1,1,1};
    RTsize mDimensionality = 1;
    Format mFormat = FLOAT;
    RTsize mFormatElements = 4;

    float * mData = nullptr;
    size_t get_number_of_elements() const { return mDimensions[0]*mDimensions[1]*mDimensions[2]*mFormatElements; }

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
        size_t element_size;
        Texture::Format format;

        archive(
                cereal::make_nvp("width", dims[0]),
                cereal::make_nvp("height", dims[1]),
                cereal::make_nvp("depth", dims[2]),
                cereal::make_nvp("dimensionality", dimensions),
                cereal::make_nvp("format", format),
                cereal::make_nvp("element_size", element_size)
        );

        construct->set_format(element_size, format);
        construct->set_size(dimensions, dims);
        float * vals = new float[construct->get_number_of_elements()];
        archive.loadBinaryValue(vals,construct->get_number_of_elements()*sizeof(float), "texture");
        construct->set_data(vals, construct->get_number_of_elements()*sizeof(float));
        delete[] vals;
    }

    RTformat get_optix_format(const RTsize& element_size, const Texture::Format& format) const;
    void get_size_and_format(const RTformat & out_format, RTsize & element_size, Texture::Format & format) const;
    size_t get_size(const Texture::Format & format) const;

};

template<>
inline void Texture::save(cereal::XMLOutputArchiveOptix & archive) const
{
    archive(
            cereal::make_nvp("width", mDimensions[0]),
            cereal::make_nvp("height", mDimensions[1]),
            cereal::make_nvp("depth", mDimensions[2]),
            cereal::make_nvp("dimensionality", mDimensionality),
            cereal::make_nvp("format", mFormat),
            cereal::make_nvp("element_size", mFormatElements)
    );
    archive.saveBinaryValue(mData, get_number_of_elements()*sizeof(float), "texture");
}


template<typename T>
T Texture::get_texel(size_t i, size_t j, size_t k) const
{
    size_t idx = (k * get_height() + j) * get_width() + i;
    size_t size = get_size(mFormat);
    char* data = reinterpret_cast<char*>(mData);
    T * res = reinterpret_cast<T*>(&data[size*mFormatElements*idx]);
    return *res;
}


template<typename T>
void Texture::set_texel(const T &val, size_t i, size_t j, size_t k)
{
    size_t idx = (k * get_height() + j) * get_width() + i;
    size_t size = get_size(mFormat);
    char* data = reinterpret_cast<char*>(mData);
    T * d = reinterpret_cast<T*>(&data[size*mFormatElements*idx]);
    *d = val;
}

