#pragma once
#include <optixu/optixpp_namespace.h>

#include <string>
#include <iosfwd>
#include "texture.h"
#include <memory>

std::unique_ptr<Texture> loadRAWTexture(optix::Context& context,
	const std::string& filename,
	const optix::float3& default_color);


class RAWLoader
{
public:
	RAWLoader(const std::string& filename);
	~RAWLoader();

	bool           failed()const;
	unsigned int   width()const;
	unsigned int   height()const;
	float*         raster()const;

private:
	unsigned int   m_nx;
	unsigned int   m_ny;
	float*         m_raster;
};
