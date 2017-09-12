#pragma once
#include <cereal/archives/json.hpp>
#include <optix_world.h>
#include "logger.h"

#include <sstream>

namespace cereal
{
	template<class Archive> void serialize(Archive & archive, optix::float1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::float2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::float3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::float4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w) ); }

	template<class Archive> void serialize(Archive & archive, optix::int1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::int2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::int3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::int4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::uint1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::uint2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::uint3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::uint4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::char1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::char2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::char3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::char4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::uchar1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::uchar2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::uchar3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::uchar4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::short1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::short2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::short3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::short4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w)); }

	template<class Archive> void serialize(Archive & archive, optix::ushort1 & m) { archive(cereal::make_nvp("x", m.x)); }
	template<class Archive> void serialize(Archive & archive, optix::ushort2 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y)); }
	template<class Archive> void serialize(Archive & archive, optix::ushort3 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z)); }
	template<class Archive> void serialize(Archive & archive, optix::ushort4 & m) { archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z), cereal::make_nvp("w", m.w)); }

}

inline void test_cereal()
{
	optix::float3 example = optix::make_float3(1, 0, 0);
	std::stringstream ss;
	{
		cereal::JSONOutputArchive oarchive(ss); // Create an output archive
		{
			oarchive(cereal::make_nvp("vec0", example));
		}
		{
			oarchive(cereal::make_nvp("vec1", optix::make_uint2(1024, 21)));
		}
		{
			oarchive(cereal::make_nvp("vec2", optix::make_uchar4('c', 'i', 'a', 'o')));
		}
		{
			oarchive(cereal::make_nvp("vec3", optix::make_short4(0,0,1,5)));
		}
	}
	

	Logger::info << ss.str();
	cereal::JSONInputArchive iarchive(ss);
	optix::float3 e1;
	optix::uint2 e2;
	optix::uchar4 e3;
	optix::short4 e4;
	iarchive(e1,e2,e3,e4);
	Logger::info << e1.x;

}