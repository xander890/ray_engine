//
// Created by alcor on 4/24/18.
//

#include <cstring>
#include "texture.h"

Texture::Texture(optix::Context &context)
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

void Texture::unmap_data()
{
    textureBuffer->unmap();
}

float* Texture::map_data()
{
    return reinterpret_cast<float*>(textureBuffer->map());
}

void Texture::set_data(float *data, size_t size)
{
    assert(size <= width*height*4*sizeof(float));
    float * dest = map_data();
    memcpy(dest, data, size);
    unmap_data();
}

void Texture::set_size(int w, int h, int d)
{
    width = w > 0? static_cast<unsigned int>(w) : 1;
    height = h > 0? static_cast<unsigned int>(h) : 1;
    depth = d > 0? static_cast<unsigned int>(d) : 1;
    int dims = 1;
    dims += h > 0? 1 : 0;
    dims += d > 0? 1 : 0;
    RTsize dimensions[3] = {(RTsize)width, (RTsize)height, (RTsize)depth};
    textureBuffer->setSize(dims, &dimensions[0]);
}

Texture::~Texture()
{
    textureSampler->destroy();
    textureBuffer->destroy();
}
