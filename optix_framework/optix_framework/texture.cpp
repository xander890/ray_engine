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
    assert(size <= width*height*4*sizeof(float));
    float * dest = reinterpret_cast<float*>(textureBuffer->map());
    memcpy(dest, data, size);
    memcpy(mData, data, size);
    textureBuffer->unmap();
}

void Texture::set_size(int w, int h, int d)
{
    width = w > 0? static_cast<unsigned int>(w) : 1;
    height = h > 0? static_cast<unsigned int>(h) : 1;
    depth = d > 0? static_cast<unsigned int>(d) : 1;
    dimensions = 1;
    dimensions += h > 0? 1 : 0;
    dimensions += d > 0? 1 : 0;
    int dims[3] = {width, height, depth};
    set_size(dimensions, dims);
}

Texture::~Texture()
{
    textureSampler->destroy();
    textureBuffer->destroy();
}

void Texture::set_size(int dimensions, int *dims)
{
    RTsize * d = new RTsize[dimensions];
    int total_size = 1;
    for(int i = 0; i < dimensions; i++)
    {
        d[i] = (RTsize) dims[i];
        total_size *= d[i];
    }
    textureBuffer->setSize(dimensions, &d[0]);
    delete[] d;
    if(mData != nullptr)
    {
        delete[] mData;
    }
    mData = new float[total_size*4];
}

optix::float4 Texture::get_texel(int i, int j, int k) const
{
    int idx = (i * get_width() + j) * get_height() + k;
    auto res = reinterpret_cast<optix::float4*>(&mData[idx * 4]);
    return *res;
}