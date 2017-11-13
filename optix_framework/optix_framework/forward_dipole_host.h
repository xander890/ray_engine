#pragma once
#include "bssrdf_host.h"

class ForwardDipole : public BSSRDF
{
public:
	ForwardDipole(optix::Context & ctx);
	void load(const ScatteringMaterialProperties & props) override {}
	void on_draw() override {}
};