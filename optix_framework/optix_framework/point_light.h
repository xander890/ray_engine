#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include <optix.h>

struct PointLight
{
  optix::float3 position;
  int padding; // For Opengl std140
  optix::float3 intensity;
  int casts_shadow; 
};

#endif // POINTLIGHT_H
