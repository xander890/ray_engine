#pragma once
#include "optix_world.h"
#include "GL/glew.h"
#include <cstring>

template<typename T> optix::Buffer create_glbo_buffer(optix::Context & ctx, unsigned int type, unsigned int size)
{
	GLuint buf;
	glCreateBuffers(1, &buf);

	GLint bind;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bind);
	glBindBuffer(GL_ARRAY_BUFFER, buf);
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, bind);
	optix::Buffer r = ctx->createBufferFromGLBO(type, buf);
	r->setSize(size);
	return r;
}

template<typename T> void resize_glbo_buffer(optix::Buffer & buffer, unsigned int size)
{
	GLuint buf = buffer->getGLBOId();

	GLint bind;
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bind);
	glBindBuffer(GL_ARRAY_BUFFER, buf);
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, bind);
	buffer->setSize(size);
}


template<typename T> optix::Buffer create_buffer(optix::Context & ctx, unsigned int type, int size)
{
	optix::Buffer buf = ctx->createBuffer(type);
	buf->setFormat(RT_FORMAT_USER);
	buf->setElementSize(sizeof(T));
	buf->setSize(size);

    return buf;
}

template<typename T> optix::Buffer create_buffer(optix::Context & ctx)
{
	return create_buffer<T>(ctx, RT_BUFFER_INPUT, 1);
}

template<typename T> void initialize_buffer(optix::Buffer & buf, std::vector<T>& data)
{
    buf->setSize(data.size());
    T * ar = reinterpret_cast<T*>(buf->map());
    memcpy(ar, data.data(), sizeof(T) * data.size());
    buf->unmap();
}

template<typename T> void fill_buffer(optix::Buffer & buf, const T & element)
{
	size_t size;
	buf->getSize(size);
	T * ar = reinterpret_cast<T*>(buf->map());
	for(int i = 0; i < size; i++)
		ar[i] = element;
	buf->unmap();
}

template<typename T> void initialize_buffer(optix::Buffer & buf, std::initializer_list<T> data)
{
    buf->setSize(data.size());
    T * ar = reinterpret_cast<T*>(buf->map());
    memcpy(ar, data.begin(), sizeof(T) * data.size());
    buf->unmap();
}

template<typename T> void initialize_buffer(optix::Buffer & buf, const T& data)
{
    initialize_buffer<T>(buf, { data });
}

template<typename T> optix::Buffer create_and_initialize_buffer(optix::Context & ctx, const T& obj)
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

inline unsigned int add_entry_point(optix::Context & ctx, optix::Program & program)
{
	unsigned int current = ctx->getEntryPointCount();
	ctx->setEntryPointCount(current + 1);
	ctx->setRayGenerationProgram(current, program);
	return current;
}

inline void clear_buffer(optix::Buffer & buf)
{
	RTsize element_size = buf->getElementSize();
	unsigned int dimensionality = buf->getDimensionality();
	RTsize dims[3];
	buf->getSize(dimensionality, &dims[0]);
	RTsize size = 1;
	for (unsigned int i = 0; i < dimensionality; i++)
		size *= dims[i];

	float* buff = reinterpret_cast<float*>(buf->map());
	memset(buff, 0, size * element_size);
	buf->unmap();
}

inline optix::Buffer clone_buffer(optix::Buffer & buf, unsigned int type)
{
	RTsize element_size = buf->getElementSize();
	unsigned int dimensionality = buf->getDimensionality();
	RTsize dims[3];
	buf->getSize(dimensionality, &dims[0]);
	RTsize size = 1;
	for (unsigned int i = 0; i < dimensionality; i++)
		size *= dims[i];

	optix::Buffer ret = buf->getContext()->createBuffer(type);

	// Format
	RTformat format = buf->getFormat();
	ret->setFormat(format);
	if(format == RT_FORMAT_USER)
		ret->setElementSize(element_size);

	// Size
	if(dimensionality == 1)
		ret->setSize(dims[0]);
	else if (dimensionality == 2)
		ret->setSize(dims[0], dims[1]);
	else if (dimensionality == 3)
		ret->setSize(dims[0], dims[1], dims[2]);

	// Mipmaps
	ret->setMipLevelCount(buf->getMipLevelCount());

	// And finally, data
	void * buf_ptr = buf->map();
	void * ret_ptr = ret->map();
	memcpy(ret_ptr, buf_ptr, size*element_size);
	buf->unmap();
	ret->unmap();
	return ret;
}

inline optix::Variable get_var(optix::ScopedObj * ptr, const char * varname)
{
	optix::Variable v = ptr->queryVariable(varname);
	if (v.operator->() == 0)
		v = ptr->declareVariable(varname);
	return v;
}