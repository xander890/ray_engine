//
// Created by alcor on 4/24/18.
//
#pragma once
#include "host_device_common.h"
#include "optix_serialize.h"
#include "logger.h"

class Thumbnail;


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
    RTformat get_format() const;
    void get_format(RTsize& elements, Texture::Format & format) const;

    size_t get_width() const { return mDimensions[0]; }
    size_t get_height() const { return mDimensions[1]; }
    size_t get_depth() const { return mDimensions[2]; }

    TexPtr get_id();

    const void* get_data() const { return mData; }


    template<typename T>
    T get_max() const
    {
        switch(mFormat)
        {
            case BYTE:
            case UNSIGNED_BYTE:
            case SHORT:
            case UNSIGNED_SHORT:
            case INT:
            case UNSIGNED_INT:
                return std::numeric_limits<T>::max();
            default:
            case FLOAT: return 1;
        }
    }

    template<typename T>
    void* get_normalized_data() const
    {
        T* data = reinterpret_cast<T*>(mData);
        T* to_return = new T[get_number_of_elements()];
        T mx = std::numeric_limits<T>::min();

        for(int i = 0; i < get_number_of_elements(); i++)
        {
            to_return[i] = data[i];
            mx = std::max(mx, data[i]);
        }

        for(int i = 0; i < get_number_of_elements(); i++)
        {
            to_return[i] = to_return[i] * get_max<T>();
        }
        return to_return;
    }

    void* get_normalized_data() const
    {
        switch(mFormat)
        {
            case BYTE: return get_normalized_data<char>();
            case UNSIGNED_BYTE: return get_normalized_data<unsigned char>();
            case SHORT: return get_normalized_data<short>();
            case UNSIGNED_SHORT: return get_normalized_data<unsigned short>();
            case INT: return get_normalized_data<int>();
            case UNSIGNED_INT: return get_normalized_data<unsigned int>();
            default:
            case FLOAT: return get_normalized_data<float>();
        }
    }


    void set_data(void * data, size_t size);
    bool on_draw();

    void* get_texel_ptr(size_t x, size_t y = 0, size_t z = 0) const;
    void set_texel_ptr(const void* val, size_t x, size_t y = 0, size_t z = 0);

    optix::TextureSampler get_sampler() { return textureSampler; }

    void update();

    static RTformat get_optix_format(const RTsize& element_size, const Texture::Format& format);
    static unsigned int get_gl_element(const RTsize& element_size);
    static unsigned int get_gl_format(const Format& element_format);
    static void get_size_and_format(const RTformat & out_format, RTsize & element_size, Texture::Format & format);
    static size_t get_size(const Texture::Format & format);

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

    optix::Context mContext;
    std::unique_ptr<Thumbnail> mThumbnail;

    template<typename T>
    void check_template_access() const { assert(sizeof(T) == get_size(mFormat)); }

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

