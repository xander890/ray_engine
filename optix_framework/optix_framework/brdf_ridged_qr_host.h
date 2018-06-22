#pragma once
#include "brdf_host.h"

/*
* Class representing a MERL material. Material is loaded into buffers via this class, then sampled at runtime.
*/
class RidgedBRDF : public BRDF
{ 
public:
	RidgedBRDF(optix::Context & ctx, BRDFType::Type type) : BRDF(ctx, type)
	{

	}
	
	// Loads the data into a specific material.
	void load(MaterialHost &obj) override;

	bool on_draw() override;

private:
	RidgedBRDF() : BRDF() {}

	// Serialization classes
	friend class cereal::access;
	void load(cereal::XMLInputArchiveOptix & archive)
	{
		archive(cereal::base_class<BRDF>(this),
			cereal::make_nvp("ridge_angle", ridge_angle)
		);
	}

	void save(cereal::XMLOutputArchiveOptix & archive) const
	{
		archive(cereal::base_class<BRDF>(this),
			cereal::make_nvp("ridge_angle", ridge_angle)
		);
	}
	float ridge_angle = 20.0f;
};

CEREAL_CLASS_VERSION(RidgedBRDF, 0)
CEREAL_REGISTER_TYPE(RidgedBRDF)
CEREAL_REGISTER_POLYMORPHIC_RELATION(BRDF, RidgedBRDF)