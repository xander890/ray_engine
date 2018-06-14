#pragma once
#include <optix_world.h>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include "cereal/types/memory.hpp"
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/cereal.hpp>
#include <xml_archive.hpp>

namespace cereal
{

	template<class Archive> void serialize(Archive & archive, optix::float1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::float2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::float3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::float4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w) ); }

	template<class Archive> void serialize(Archive & archive, optix::int1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::int2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::int3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::int4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::uint1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::uint2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::uint3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::uint4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::char1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::char2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::char3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::char4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::uchar1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::uchar2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::uchar3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::uchar4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::short1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::short2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::short3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::short4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::ushort1 & m) { archive(make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::ushort2 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::ushort3 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::ushort4 & m) { archive(make_nvp("x", m.x), make_nvp("y", m.y), make_nvp("z", m.z), make_nvp("w", m.w)); }

}

inline void save_buffer(cereal::XMLOutputArchiveOptix & archive, optix::Buffer buffer, std::string name)
{
	void * data = buffer->map();
	unsigned int dim = buffer->getDimensionality();
	std::vector<RTsize> dims = std::vector<RTsize>(dim);
	buffer->getSize(dim, &dims[0]);
	RTsize total_size = 1;
	for(unsigned int i = 0; i < dim; i++)
		total_size *= dims[i];

	RTsize element = buffer->getElementSize();
	archive.saveBinaryValue(data, total_size * element, name.c_str());
	buffer->unmap();
	archive(cereal::make_nvp(name + "_element_size", element));
	archive(cereal::make_nvp(name + "_dimensionality", dim));
	archive(cereal::make_nvp(name + "_size", dims));
}

inline void load_buffer(cereal::XMLInputArchiveOptix & archive, optix::Buffer & buffer, std::string name)
{
	RTsize element, dim;
	archive(cereal::make_nvp(name + "_element_size", element));
	archive(cereal::make_nvp(name + "_dimensionality", dim));

	std::vector<RTsize> dims = std::vector<RTsize>(dim);
	archive(cereal::make_nvp(name + "_size", dims));

	buffer = archive.get_context()->createBuffer(RT_BUFFER_INPUT);
	buffer->setFormat(RT_FORMAT_USER);
	buffer->setSize((unsigned int)dim, &dims[0]);
	buffer->setElementSize(element);

	RTsize total_size = 1;
	for(int i = 0; i < dim; i++)
		total_size *= dims[i];

	void * data = buffer->map();
	archive.loadBinaryValue(data, total_size * element, name.c_str());
	buffer->unmap();
}