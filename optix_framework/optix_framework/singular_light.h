// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#ifndef DIRECTIONAL_H
#define DIRECTIONAL_H

#include <optix_math.h>

struct SingularLightData
{
  optix::float3 direction;
  int type;
  optix::float3 emission;
  int casts_shadow; 
};

#endif // DIRECTIONAL_H