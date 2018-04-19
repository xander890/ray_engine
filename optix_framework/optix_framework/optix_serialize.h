#pragma once
#include <optix_world.h>
#include "logger.h"
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include "cereal/types/memory.hpp"
#include <xml_archive.hpp>
#include <cereal/archives/json.hpp>

namespace cereal
{

    using XMLOutputArchive = XMLOutputArchiveOptix;

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

