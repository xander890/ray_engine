#pragma once
#include <memory>
#include <host_device_common.h>
#include "bssrdf_properties.h"
#include "mesh.h"

class BSSRDF : std::enable_shared_from_this<BSSRDF>
{
public:
	BSSRDF(optix::Context & ctx, ScatteringDipole::Type);
	virtual bool on_draw();
	virtual void load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &props);
	virtual ScatteringDipole::Type get_type() { return mType; };

    virtual optix::float3 get_sampling_inverse_mean_free_path(const ScatteringMaterialProperties &props) { return props.transport; }

	static std::unique_ptr<BSSRDF> create(optix::Context & ctx, ScatteringDipole::Type type);
	static bool dipole_selector_gui(ScatteringDipole::Type & type, std::string id = "");
protected:
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

