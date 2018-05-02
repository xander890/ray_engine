//
// Created by alcor on 4/24/18.
//

#include <cstring>
#include "texture.h"

Texture::Texture(optix::Context context)
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
    textureBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1u, 1u);
    textureSampler->setBuffer(0u, 0u, textureBuffer);
}

TexPtr Texture::get_id()
{
    return textureSampler->getId();
}

void Texture::set_data(float *data, size_t size)
{
    assert(size <= get_number_of_elements()*sizeof(float));
    float * dest = reinterpret_cast<float*>(textureBuffer->map());
    memcpy(dest, data, size);
    memcpy(mData, data, size);
    textureBuffer->unmap();
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
    textureSampler->destroy();
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

optix::float4 Texture::get_texel(size_t i, size_t j, size_t k) const
{
    size_t idx = (i * get_width() + j) * get_height() + k;
    auto res = reinterpret_cast<optix::float4*>(&mData[idx * 4]);
    return *res;
}
