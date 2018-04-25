//
// Created by alcor on 4/24/18.
//
#pragma once
#include "host_device_common.h"

class Texture
{
public:
    Texture(optix::Context & context);
    ~Texture();
    void set_size(int width, int height = -1, int depth = -1);

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
};


