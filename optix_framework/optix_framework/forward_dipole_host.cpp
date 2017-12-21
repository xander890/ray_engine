#include "forward_dipole_host.h"

ForwardDipole::ForwardDipole(optix::Context & context): BSSRDF(context, ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF)
{
	std::string ptx_path_output = get_path_ptx("forward_dipole.cu");
	static optix::Program ray_gen_program_sample = context->createProgramFromPTXFile(ptx_path_output, "sampleLengthDipoleProgram");
	static optix::Program ray_gen_program_monopole = context->createProgramFromPTXFile(ptx_path_output, "evalMonopoleProgram");
	static optix::Program ray_gen_program_dipole = context->createProgramFromPTXFile(ptx_path_output, "evalDipoleProgram");

	context["sampleLengthDipole"]->setInt(ray_gen_program_sample->getId());
	context["evalMonopole"]->setInt(ray_gen_program_monopole->getId());
	context["evalDipole"]->setInt(ray_gen_program_dipole->getId());
}  
