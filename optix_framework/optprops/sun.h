#ifndef SUN_H
#define SUN_H

#include "CGLA/Vec3f.h"
#include "CGLA/Vec2f.h"
#include "Medium.h"

OPTPROPS_API CGLA::Vec2f sun_position(double day, double time, double latitude);
OPTPROPS_API Medium mean_solar_irrad();
OPTPROPS_API Medium solar_irrad(double day, double time, double latitude, const CGLA::Vec3f& up, CGLA::Vec3f& direction);
OPTPROPS_API Medium direct_sun(double day, double time, double latitude, const CGLA::Vec3f& up, CGLA::Vec3f& direction);
OPTPROPS_API Medium atmosphere(double day, double time, double latitude, const CGLA::Vec3f& up, CGLA::Vec3f& direction);

#endif // SUN_H
