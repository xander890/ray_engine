// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#ifndef DIRECTIONAL_H
#define DIRECTIONAL_H

#include <optix_math.h>
#include "host_device_common.h"
struct SingularLightData
{
  optix::float3 direction   DEFAULT(optix::make_float3(0,-1,0));
  LightType::Type type      DEFAULT(LightType::DIRECTIONAL);
  optix::float3 emission    DEFAULT(optix::make_float3(1,1,1));
  int casts_shadow          DEFAULT(1);
};

#endif // DIRECTIONAL_H