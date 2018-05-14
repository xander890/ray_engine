#pragma once
#include "bssrdf_host.h"

class ForwardDipole : public BSSRDF
{
public:
	ForwardDipole(optix::Context & ctx);
	void load(const float relative_ior, const ScatteringMaterialProperties &props) override {}
	bool on_draw() override { return false;  }

private:
	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(
				cereal::base_class<BSSRDF>(this)
		);
	}
};