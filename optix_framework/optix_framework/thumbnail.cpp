//
// Created by alcor on 6/6/18.
//

#include "thumbnail.h"
#include "GL/gl.h"
#include <algorithm>
#include <climits>

Thumbnail::Thumbnail(optix::Context ctx, const Texture &texture, optix::int2 size = optix::make_int2(100,100))
{
    glGenTextures( 1, &mId );
    glBindTexture( GL_TEXTURE_2D, mId);

    // Change these to GL_LINEAR for super- or sub-sampling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    RTsize elements;
    Texture::Format format;
    texture.get_format(elements, format);
    const void * data = texture.get_normalized_data();

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (int)texture.get_width(), (int)texture.get_height(),
            0, Texture::get_gl_element(elements), Texture::get_gl_format(format), data);

    glBindTexture( GL_TEXTURE_2D, 0);
}

Thumbnail::~Thumbnail()
{
    glDeleteTextures(1, &mId);
}
