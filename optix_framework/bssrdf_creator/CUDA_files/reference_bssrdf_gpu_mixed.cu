// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <photon_trace_reference_bssrdf.h>
#include <md5.h>
#include <material.h>
#include <photon_trace_structs.h>

using namespace optix;

rtDeclareVariable(BufPtr2D<float>, reference_resulting_flux_1, ,);
rtDeclareVariable(BufPtr2D<float>, reference_resulting_flux_2, ,);
rtDeclareVariable(BufPtr2D<float>, reference_resulting_flux_total, ,);

RT_PROGRAM void post()
{
    float connection_result = reference_resulting_flux_1[launch_index];
    float mcml_result = reference_resulting_flux_2[launch_index];
    reference_resulting_flux_total[launch_index] = connection_result + mcml_result;
}



