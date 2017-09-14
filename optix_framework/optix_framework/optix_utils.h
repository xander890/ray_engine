#pragma once
#include "optix_world.h"

template<typename T> optix::Buffer create_buffer(optix::Context & ctx)
{
    optix::Buffer buf = ctx->createBuffer(RT_BUFFER_INPUT);
    buf->setFormat(RT_FORMAT_USER);
    buf->setElementSize(sizeof(T));
    buf->setSize(1);
    return buf;
}

template<typename T> void initialize_buffer(optix::Buffer & buf, std::vector<T>& data)
{
    buf->setSize(data.size());
    T * ar = reinterpret_cast<T*>(buf->map());
    memcpy(ar, data.data(), sizeof(T) * data.size());
    buf->unmap();
}

template<typename T> void initialize_buffer(optix::Buffer & buf, std::initializer_list<T> data)
{
    buf->setSize(data.size());
    T * ar = reinterpret_cast<T*>(buf->map());
    memcpy(ar, data.begin(), sizeof(T) * data.size());
    buf->unmap();
}

template<typename T> void initialize_buffer(optix::Buffer & buf, T& data)
{
    initialize_buffer<T>(buf, { data });
}

template<typename T> optix::Buffer create_and_initialize_buffer(optix::Context & ctx, T& obj)
{
    optix::Buffer buf = create_buffer<T>(ctx);
    initialize_buffer(buf, obj);
    return buf;
}

template<typename T> optix::Buffer create_and_initialize_buffer(optix::Context & ctx, std::initializer_list<T> objs)
{
    optix::Buffer buf = create_buffer<T>(ctx);
    buf->setSize(objs.size());
    initialize_buffer(buf, objs);
    return buf;
}

template<typename T> optix::Buffer create_and_initialize_buffer(optix::Context & ctx, std::vector<T>& objs)
{
    optix::Buffer buf = create_buffer<T>(ctx);
    initialize_buffer(buf, objs);
    return buf;
}

template<typename T> void get_texture_pixel(optix::Context & ctx, T&elem, int texture_ptr)
{
	if (texture_ptr < 0)
		return;
	optix::TextureSampler tex = ctx->getTextureSamplerFromId(texture_ptr);
	optix::Buffer buf = tex->getBuffer();
	assert(buf->getElementSize() == sizeof(T));
	memcpy(&elem, buf->map(), sizeof(T));
	buf->unmap();
}

template<typename T> void set_texture_pixel(optix::Context & ctx, const T& elem, int texture_ptr)
{
	if (texture_ptr < 0)
		return;
	optix::TextureSampler tex = ctx->getTextureSamplerFromId(texture_ptr);
	optix::Buffer buf = tex->getBuffer();
	buf->setSize(1);
	assert(buf->getElementSize() == sizeof(T));
	memcpy(buf->map(), &elem , sizeof(T));
	buf->unmap();
}