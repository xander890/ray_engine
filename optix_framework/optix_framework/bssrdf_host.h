#pragma once
#include "host_device_common.h"
#include "bssrdf_common.h"
#include "scattering_properties.h"
#include "optix_serialize_utils.h"
#include <memory>

/*
	Standard interface for BSSRDFs. Extend this interface to implement any BSSRDF model (analytical or empirical).
	BSSRDFs cannot be created directly. Use the factory method create to generate BSSRDFs according to type.
	Each BSSRDF is loaded at runtime for the specific material with the load function. Scattering properties can be used to precalculate particular options for the specific BSSRDF.
*/
class BSSRDF : std::enable_shared_from_this<BSSRDF>
{
public:
	static std::unique_ptr<BSSRDF> create(optix::Context & ctx, ScatteringDipole::Type type);

	virtual bool on_draw();
	virtual void load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &props);
	virtual ScatteringDipole::Type get_type();
	virtual optix::float3 get_sampling_inverse_mean_free_path(const ScatteringMaterialProperties &props);
	static bool bssrdf_selection_gui(ScatteringDipole::Type & type, std::string id = "");

protected:
	BSSRDF(optix::Context & ctx, ScatteringDipole::Type);
	optix::Context mContext;
	ScatteringDipole::Type mType;

	friend class cereal::access;
	template<class Archive>
	void load(Archive & archive)
	{
		std::string s;
		archive(cereal::make_nvp("type", s));
		mType = ScatteringDipole::to_enum(s);
	}

	template<class Archive>
	void save(Archive & archive) const
	{
		archive(cereal::make_nvp("type", ScatteringDipole::to_string(mType)));
	}

};

