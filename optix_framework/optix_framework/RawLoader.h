#pragma once
#include <optixu/optixpp_namespace.h>

#include <string>
#include <iosfwd>
#include "texture.h"
#include <memory>

bool loadRAWTexture(std::unique_ptr<Texture> &return_tex, optix::Context &context, const std::string &filename);


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
