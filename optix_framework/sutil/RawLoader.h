#pragma once
#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <string>
#include <iosfwd>

SUTILAPI optix::TextureSampler loadRAWTexture(optix::Context& context,
	const std::string& filename,
	const optix::float3& default_color);


class RAWLoader
{
public:
	SUTILAPI RAWLoader(const std::string& filename);
	SUTILAPI ~RAWLoader();

	SUTILAPI bool           failed()const;
	SUTILAPI unsigned int   width()const;
	SUTILAPI unsigned int   height()const;
	SUTILAPI float*         raster()const;

private:
	unsigned int   m_nx;
	unsigned int   m_ny;
	float*         m_raster;
};
