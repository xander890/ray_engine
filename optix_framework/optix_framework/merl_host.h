#pragma once
#include "brdf_host.h"

/*
* Class representing a MERL material. Material is loaded into buffers via this class, then sampled at runtime.
*/
class MERLBRDF : public BRDF
{
public:
	MERLBRDF(optix::Context & ctx, BRDFType::Type type) : BRDF(ctx, type)
	{

	}
	MERLBRDF(const MERLBRDF& other);
	MERLBRDF& operator=(const MERLBRDF& other);

	~MERLBRDF();

	// Loads data from a .binary MERL brdf file into the class.
	void load_brdf_file(std::string file);

	// Loads the data into a specific material.
	void load(MaterialHost &obj) override;

	bool on_draw() override;

private:
	void init();
	MERLBRDF() : BRDF() {}

	// Serialization classes
	friend class cereal::access;
	void load(cereal::XMLInputArchiveOptix & archive)
	{
		int size;
		archive(cereal::base_class<BRDF>(this),
			cereal::make_nvp("name", mName),
			cereal::make_nvp("reflectance", mReflectance),
			cereal::make_nvp("merl_correction", mCorrection),
			cereal::make_nvp("merl_size", size)
		);
		mData.resize(size);
		archive.loadBinaryValue(mData.data(), size * sizeof(float), "merl_data");
	}

	void save(cereal::XMLOutputArchiveOptix & archive) const
	{
		archive(cereal::base_class<BRDF>(this),
			cereal::make_nvp("name", mName),
			cereal::make_nvp("reflectance", mReflectance),
			cereal::make_nvp("merl_correction", mCorrection),
			cereal::make_nvp("merl_size", mData.size())
		);

		archive.saveBinaryValue(mData.data(), mData.size() * sizeof(float), "merl_data");
	}

	bool mInit = false;
	optix::float3 mCorrection = optix::make_float3(1);
	std::vector<float> mData;
	optix::float3 mReflectance = optix::make_float3(0);
	optix::Buffer mMerlBuffer = nullptr;
	std::string mName = "";
};

CEREAL_CLASS_VERSION(MERLBRDF, 0)
CEREAL_REGISTER_TYPE(MERLBRDF)
CEREAL_REGISTER_POLYMORPHIC_RELATION(BRDF, MERLBRDF)