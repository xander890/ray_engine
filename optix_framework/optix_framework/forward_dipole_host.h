#pragma once
#include "bssrdf_host.h"

class ForwardDipole : public BSSRDF
{
public:
	ForwardDipole(optix::Context & ctx) : BSSRDF(ctx, ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF) {}
};