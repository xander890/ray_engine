#pragma once
#include "host_device_common.h"
#include "texture.h"
#include <memory>


inline std::unique_ptr<Texture> resize(optix::Context ctx, const Texture& texture, optix::int2 new_size)
{
    std::unique_ptr<Texture> t = std::make_unique<Texture>(ctx);
    t->set_size(new_size.x, new_size.y);
    t->set_format(texture.get_format());

    for(int i = 0; i < t->get_width(); i++)
    {
        for(int j = 0; j < t->get_height(); j++)
        {
            int new_i = (int)(i * texture.get_width() / ((float)new_size.x) );
            int new_j = (int)(j * texture.get_height() / ((float)new_size.y) );

            auto value = texture.get_texel_ptr(new_i,new_j);
            t->set_texel_ptr(value, i, j);
        }
    }
    return t;
}

class Thumbnail
{
public:
    Thumbnail(optix::Context ctx, const Texture &texture, optix::int2 size);
    ~Thumbnail();
    unsigned int get_id() { return mId; }
private:
    unsigned int mId;
};