#include "forward_dipole_host.h"

ForwardDipole::ForwardDipole(optix::Context & context): BSSRDF(context, ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF)
{

}  
