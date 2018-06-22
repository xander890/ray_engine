//
// Created by alcor on 4/24/18.
//

#include <cstring>
#include "GL/glew.h"
#include "texture.h"
#include "immediate_gui.h"
#include "thumbnail.h"

RTformat Texture::get_optix_format(const RTsize& element_size, const Texture::Format& format)
{
    assert(element_size > 0 && element_size <= 4);
    switch(format)
    {
        case Texture::Format::BYTE:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_BYTE;
                case 2: return RT_FORMAT_BYTE2;
                case 3: return RT_FORMAT_BYTE3;
                case 4: return RT_FORMAT_BYTE4;
            }
        }
        case Texture::Format::UNSIGNED_BYTE:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_UNSIGNED_BYTE;
                case 2: return RT_FORMAT_UNSIGNED_BYTE2;
                case 3: return RT_FORMAT_UNSIGNED_BYTE3;
                case 4: return RT_FORMAT_UNSIGNED_BYTE4;
            }
        }
        case Texture::Format::SHORT:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_SHORT;
                case 2: return RT_FORMAT_SHORT2;
                case 3: return RT_FORMAT_SHORT3;
                case 4: return RT_FORMAT_SHORT4;
            }
        }
        case Texture::Format::UNSIGNED_SHORT:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_UNSIGNED_SHORT;
                case 2: return RT_FORMAT_UNSIGNED_SHORT2;
                case 3: return RT_FORMAT_UNSIGNED_SHORT3;
                case 4: return RT_FORMAT_UNSIGNED_SHORT4;
            }
        }
        case Texture::Format::INT:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_INT;
                case 2: return RT_FORMAT_INT2;
                case 3: return RT_FORMAT_INT3;
                case 4: return RT_FORMAT_INT4;
            }
        }
        case Texture::Format::UNSIGNED_INT:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_UNSIGNED_INT;
                case 2: return RT_FORMAT_UNSIGNED_INT2;
                case 3: return RT_FORMAT_UNSIGNED_INT3;
                case 4: return RT_FORMAT_UNSIGNED_INT4;
            }
        }
        default:
        case Texture::Format::FLOAT:
        {
            switch(element_size)
            {
                case 1: return RT_FORMAT_FLOAT;
                case 2: return RT_FORMAT_FLOAT2;
                case 3: return RT_FORMAT_FLOAT3;
                case 4: return RT_FORMAT_FLOAT4;
            }
        }

    }
	return RT_FORMAT_USER;
}

void Texture::get_size_and_format(const RTformat & out_format, RTsize & element_size, Texture::Format & format)
{
   switch(out_format)
   {
       case RT_FORMAT_FLOAT :                   format = Texture::Format::FLOAT; element_size = 1; break;
               case RT_FORMAT_FLOAT2 :          format = Texture::Format::FLOAT; element_size = 2; break;
               case RT_FORMAT_FLOAT3 :          format = Texture::Format::FLOAT; element_size = 3; break;
               case RT_FORMAT_FLOAT4 :          format = Texture::Format::FLOAT; element_size = 4; break;
               case RT_FORMAT_BYTE :            format = Texture::Format::BYTE; element_size = 1; break;
               case RT_FORMAT_BYTE2 :           format = Texture::Format::BYTE; element_size = 2; break;
               case RT_FORMAT_BYTE3 :           format = Texture::Format::BYTE; element_size = 3; break;
               case RT_FORMAT_BYTE4 :           format = Texture::Format::BYTE; element_size = 4; break;
               case RT_FORMAT_UNSIGNED_BYTE :   format = Texture::Format::UNSIGNED_BYTE; element_size = 1; break;
               case RT_FORMAT_UNSIGNED_BYTE2 :  format = Texture::Format::UNSIGNED_BYTE; element_size = 2; break;
               case RT_FORMAT_UNSIGNED_BYTE3 :  format = Texture::Format::UNSIGNED_BYTE; element_size = 3; break;
               case RT_FORMAT_UNSIGNED_BYTE4 :  format = Texture::Format::UNSIGNED_BYTE; element_size = 4; break;
               case RT_FORMAT_SHORT :           format = Texture::Format::SHORT; element_size = 1; break;
               case RT_FORMAT_SHORT2 :          format = Texture::Format::SHORT; element_size = 2; break;
               case RT_FORMAT_SHORT3 :          format = Texture::Format::SHORT; element_size = 3; break;
               case RT_FORMAT_SHORT4 :          format = Texture::Format::SHORT; element_size = 4; break;
               case RT_FORMAT_UNSIGNED_SHORT :  format = Texture::Format::UNSIGNED_SHORT; element_size = 1; break;
               case RT_FORMAT_UNSIGNED_SHORT2 : format = Texture::Format::UNSIGNED_SHORT; element_size = 2; break;
               case RT_FORMAT_UNSIGNED_SHORT3 : format = Texture::Format::UNSIGNED_SHORT; element_size = 3; break;
               case RT_FORMAT_UNSIGNED_SHORT4 : format = Texture::Format::UNSIGNED_SHORT; element_size = 4; break;
               case RT_FORMAT_INT    :          format = Texture::Format::INT; element_size = 1; break;
               case RT_FORMAT_INT2  :           format = Texture::Format::INT; element_size = 2; break;
               case RT_FORMAT_INT3 :            format = Texture::Format::INT; element_size = 3; break;
               case RT_FORMAT_INT4  :           format = Texture::Format::INT; element_size = 4; break;
               case RT_FORMAT_UNSIGNED_INT  :   format = Texture::Format::UNSIGNED_INT; element_size = 1; break;
               case RT_FORMAT_UNSIGNED_INT2 :   format = Texture::Format::UNSIGNED_INT; element_size = 2; break;
               case RT_FORMAT_UNSIGNED_INT3  :  format = Texture::Format::UNSIGNED_INT; element_size = 3; break;
               case RT_FORMAT_UNSIGNED_INT4 :   format = Texture::Format::UNSIGNED_INT; element_size = 4; break;
   }
}

size_t Texture::get_size(const Texture::Format & format)
{
    switch(format)
    {
        case Texture::Format::BYTE:
            return sizeof(char);
        case Texture::Format::UNSIGNED_BYTE:
            return sizeof(unsigned char);
        case Texture::Format::SHORT:
            return sizeof(short);
        case Texture::Format::UNSIGNED_SHORT:
            return sizeof(unsigned short);
        case Texture::Format::INT:
            return sizeof(int);
        case Texture::Format::UNSIGNED_INT:
            return sizeof(unsigned int);
        case Texture::Format::FLOAT:
        default:
            return sizeof(float);
    }
}

size_t Texture::get_number_of_elements(const RTformat & out_format)
{
	size_t size; Format format;
	get_size_and_format(out_format, size, format);
	return size;
}

TexPtr Texture::get_id()
{
    return mID;
}

void Texture::set_data(void *data, size_t size)
{
    assert(size <= get_memory_footprint());
    memcpy(mData, data, size);
    update();

    mThumbnail.reset();
}

void Texture::set_size(size_t w, size_t h, size_t d)
{
    mDimensions[0] = w > 0? static_cast<unsigned int>(w) : 1;
    mDimensions[1] = h > 0? static_cast<unsigned int>(h) : 1;
    mDimensions[2] = d > 0? static_cast<unsigned int>(d) : 1;
    mDimensionality = 1;
    mDimensionality += h > 0? 1 : 0;
    mDimensionality += d > 0? 1 : 0;
    set_size(mDimensionality, mDimensions);
}

Texture::~Texture()
{
    if(textureSampler.get() != nullptr)
        textureSampler->destroy();
    if(textureBuffer.get() != nullptr)
        textureBuffer->destroy();
}

void Texture::set_size(size_t dimensions, size_t *dims)
{
    mDimensionality = dimensions;
    for(int i = 0; i < 3; i++)
    {
        mDimensions[i] = (i < dimensions)? dims[i] : 1;
    }
    textureBuffer->setSize(mDimensionality, &mDimensions[0]);

    if(mData != nullptr)
    {
        delete[] mData;
    }
    mData = new float[get_number_of_elements()];

}

void Texture::set_format(const RTformat &format)
{
    RTsize elements;
    Texture::Format f;
    get_size_and_format(format, elements, f);
    set_format(elements, f);
}

void Texture::set_format(RTsize elements, const Texture::Format &format)
{
    mFormatElements = elements;
    mFormat = format;

    RTformat f = get_optix_format(elements, format);
    textureBuffer->setFormat(f);
    set_size(mDimensionality, &mDimensions[0]);
}

Texture::Texture(optix::Context context, const Texture::Format &format, const RTsize &element_size)
{
    textureSampler = context->createTextureSampler();
    textureSampler->setWrapMode(0, RT_WRAP_REPEAT);
    textureSampler->setWrapMode(1, RT_WRAP_REPEAT);
    textureSampler->setWrapMode(2, RT_WRAP_REPEAT);
    textureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    textureSampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    textureSampler->setMaxAnisotropy(1.0f);
    textureSampler->setMipLevelCount(1u);
    textureSampler->setArraySize(1u);

    textureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    textureBuffer = context->createBuffer(RT_BUFFER_INPUT);
    textureSampler->setBuffer(0u, 0u, textureBuffer);
    mID = textureSampler->getId();
    mContext = context;
    set_format(element_size, format);
    set_size(1);
}

void Texture::update()
{
    void * dest = textureBuffer->map();
    memcpy(dest, mData, get_number_of_elements() * get_size(mFormat));
    textureBuffer->unmap();
}

RTformat Texture::get_format() const
{
    return get_optix_format(mFormatElements, mFormat);
}

void Texture::get_format(RTsize &elements, Texture::Format &format) const
{
    elements = mFormatElements;
    format = mFormat;
}

bool Texture::on_draw()
{
    if(mThumbnail == nullptr)
    {
        mThumbnail = std::make_unique<Thumbnail>(mContext, *this, optix::make_int2(150,150));
    }

    unsigned int id = mThumbnail->get_id();
    ImmediateGUIDraw::Image((void *)(intptr_t)id, ImVec2(150, 150), ImVec2(0,0), ImVec2(1,1), ImVec4(1,1,1,1), ImVec4(1,1,1,1));
    ImmediateGUIDraw::SameLine();
    std::string dims;
    dims += "width: " + std::to_string(mDimensions[0]) + "\n";
    dims += (mDimensionality > 1? "height: " + std::to_string(mDimensions[1]) : "") + "\n";
    dims += (mDimensionality > 2? "depth: " + std::to_string(mDimensions[2]) : "") + "\n";
    ImmediateGUIDraw::Text(dims.c_str());
    return false;
}

void Texture::set_texel_ptr(const void *val, size_t i, size_t j, size_t k)
{
    size_t idx = (k * get_height() + j) * get_width() + i;
    size_t size = get_size(mFormat);
    char* data = reinterpret_cast<char*>(mData);
    memcpy(&data[size*mFormatElements*idx], val, size*mFormatElements);
}

void *Texture::get_texel_ptr(size_t i, size_t j, size_t k) const
{
    size_t idx = (k * get_height() + j) * get_width() + i;
    size_t size = get_size(mFormat);
    char* data = reinterpret_cast<char*>(mData);
    return &data[size*mFormatElements*idx];
}

unsigned int Texture::get_gl_element(const RTsize &element_size)
{
    switch(element_size)
    {
        case 1: return GL_RED;
        case 2: return GL_RG;
        case 3: return GL_RGB;
        case 4: return GL_RGBA;
    }
	return 0;
}

unsigned int Texture::get_gl_format(const Texture::Format &format)
{
    switch(format)
    {
        case Texture::Format::BYTE: return GL_BYTE;
        case Texture::Format::UNSIGNED_BYTE: return GL_UNSIGNED_BYTE;
        case Texture::Format::SHORT: return GL_SHORT;
        case Texture::Format::UNSIGNED_SHORT: return GL_UNSIGNED_SHORT;
        case Texture::Format::INT: return GL_INT;
        case Texture::Format::UNSIGNED_INT: return GL_UNSIGNED_INT;
        default:
        case Texture::Format::FLOAT: return GL_FLOAT;
    }
}

Texture::Texture(Texture && tex)
{
    mID = tex.mID;
    textureBuffer = tex.textureBuffer;
    textureSampler = tex.textureSampler;
    tex.textureBuffer =  nullptr;
    tex.textureSampler = nullptr;

    mContext = tex.mContext;
    mThumbnail = std::move(tex.mThumbnail);
    set_size(tex.mDimensionality, tex.mDimensions);
    set_format(tex.mFormatElements, tex.mFormat);
    set_data(tex.mData, tex.get_memory_footprint());
    delete[] tex.mData;
}
