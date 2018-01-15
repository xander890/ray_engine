#pragma once
#include <memory>
#include <host_device_common.h>
#include "bssrdf_properties.h"
#include "mesh.h"

class BSSRDF : std::enable_shared_from_this<BSSRDF>
{
public:
	BSSRDF(optix::Context & ctx, ScatteringDipole::Type);
	virtual void on_draw();
	virtual void load(const float relative_ior, const ScatteringMaterialProperties &props);
	virtual const ScatteringDipole::Type& get_type() { return mType; };

	static std::unique_ptr<BSSRDF> create(optix::Context & ctx, ScatteringDipole::Type type);
	static bool dipole_selector_gui(ScatteringDipole::Type & type, std::string id = "");
protected:
	optix::Context mContext;
	ScatteringDipole::Type mType;
};

