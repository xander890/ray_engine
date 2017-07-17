// 02576 Rendering Framework
// Written by Jeppe Revall Frisvad, 2012
// Copyright (c) DTU Informatics 2012

#ifndef SAMPLER_H
#define SAMPLER_H

#include <cmath>
#include "CGLA/Vec3f.h"
#include "CGLA/Quatf.h"
#include "mt_random.h"

// Given a direction vector v sampled on the hemisphere
// over a surface point with the z-axis as its normal,
// this function applies the same rotation to v as is
// needed to rotate the z-axis to the actual normal
// [Frisvad, Journal of Graphics Tools 16, 2012].
inline void rotate_to_normal(const CGLA::Vec3f& normal, CGLA::Vec3f& v)
{
	if(normal[2] < -0.999999f)
  {
    v = CGLA::Vec3f(-v[1], -v[0], -v[2]);
    return;
  }
  const float a = 1.0f/(1.0f + normal[2]);
  const float b = -normal[0]*normal[1]*a;
  v =   CGLA::Vec3f(1.0f - normal[0]*normal[0]*a, b, -normal[0])*v[0] 
      + CGLA::Vec3f(b, 1.0f - normal[1]*normal[1]*a, -normal[1])*v[1] 
      + normal*v[2];
}

// Given spherical coordinates, where theta is the 
// polar angle and phi is the azimuthal angle, this
// function returns the corresponding direction vector
inline CGLA::Vec3f spherical_direction(double sin_theta, double cos_theta, double phi)
{
  return CGLA::Vec3f(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
}

inline CGLA::Vec3f sample_hemisphere(const CGLA::Vec3f& normal)
{
  // Get random numbers
  double cos_theta = mt_random();
	double phi = 2.0*M_PI*mt_random();

	// Calculate new direction as if the z-axis were the normal
  double sin_theta = std::sqrt(std::max(1.0 - cos_theta*cos_theta, 0.0));
  CGLA::Vec3f v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;  
}

inline CGLA::Vec3f sample_cosine_weighted(const CGLA::Vec3f& normal)
{
  // Get random numbers
  double cos_theta = std::sqrt(mt_random());
	double phi = 2.0*M_PI*mt_random();

	// Calculate new direction as if the z-axis were the normal
  double sin_theta = std::sqrt(std::max(1.0 - cos_theta*cos_theta, 0.0));
  CGLA::Vec3f v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;
}

inline CGLA::Vec3f sample_Phong_distribution(const CGLA::Vec3f& normal, const CGLA::Vec3f& dir, float shininess)
{
  // Get random numbers
  double cos_theta = pow(mt_random(), 1.0/(shininess + 1.0));
	double phi = 2.0*M_PI*mt_random();

	// Calculate sampled direction as if the z-axis were the reflected direction
  double sin_theta = std::sqrt(std::max(1.0 - cos_theta*cos_theta, 0.0));
	CGLA::Vec3f v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual reflected direction
  rotate_to_normal(2.0*dot(normal, dir)*normal - dir, v);
  return v;
}

inline CGLA::Vec3f sample_Blinn_distribution(const CGLA::Vec3f& normal, const CGLA::Vec3f& dir, float shininess)
{
  // Get random numbers
  double cos_theta = pow(mt_random(), 1.0/(shininess + 1.0));
	double phi = 2.0*M_PI*mt_random();

	// Calculate sampled half-angle vector as if the z-axis were the normal
  double sin_theta = std::sqrt(std::max(1.0 - cos_theta*cos_theta, 0.0));
	CGLA::Vec3f hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);

  // Make sure that the half-angle vector points in the right direction
  float test = dot(hv, dir);
  if(test < 0.0f)
  {
    hv = -hv;
    test = -test;
  }

  // Return the reflection of "dir" around the half-angle vector
  return 2.0f*test*hv - dir;
}

inline CGLA::Vec3f sample_isotropic()
{
  // Get random numbers
  double cos_theta = 1.0 - 2.0*mt_random();
	double phi = 2.0*M_PI*mt_random();

	// Calculate new direction using spherical coordinates
  double sin_theta = std::sqrt(std::max(1.0 - cos_theta*cos_theta, 0.0));
  return spherical_direction(sin_theta, cos_theta, phi);
}

inline CGLA::Vec3f sample_Rayleigh(const CGLA::Vec3f& forward)
{
  // Sample spherical coordinates using rejection sampling
  double cos_theta, cos_theta_sqr;
  do
  {
    cos_theta = 1.0 - 2.0*mt_random();
    cos_theta_sqr = cos_theta*cos_theta;
  }
  while(mt_random() > 0.5*(1.0 + cos_theta_sqr));
	double phi = 2.0*M_PI*mt_random();

	// Calculate new direction using spherical coordinates
  double sin_theta = std::sqrt(std::max(1.0 - cos_theta_sqr, 0.0));
  CGLA::Vec3f v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to forward direction
  rotate_to_normal(forward, v);
  return normalize(v);
}

inline CGLA::Vec3f sample_HG(const CGLA::Vec3f& forward, double g)
{
  // Get random numbers
  double cos_theta;
  if(fabs(g) < 1.0e-3)
    cos_theta = 1.0 - 2.0*mt_random();
  else
  {
    double two_g = 2.0*g;
    double g_sqr = g*g;
    double tmp = (1.0 - g_sqr)/(1.0 - g + two_g*mt_random());
    cos_theta = 1.0/two_g*(1.0 + g_sqr - tmp*tmp);
  }
  double phi = 2.0*M_PI*mt_random();

	// Calculate new direction as if the z-axis were the forward direction
  double sin_theta = sqrt(std::max(1.0 - cos_theta*cos_theta, 0.0));
  CGLA::Vec3f v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to forward direction
  rotate_to_normal(forward, v);
  return normalize(v);
}

inline CGLA::Vec3f sample_disk(const CGLA::Vec3f& center, const CGLA::Vec3f& normal, const CGLA::Vec3f& tangent, double radius)
{
  // Get random numbers
  double r = sqrt(mt_random());
  double phi = 2.0*M_PI*mt_random();

  // Calculate displacement by rotating tangent to new position
  CGLA::Quatf qrot;
  qrot.make_rot(phi, normal);
  CGLA::Vec3f displacement = qrot.apply_unit(tangent)*(r*radius);

  // Return sampled position
  return center + displacement;
}

inline CGLA::Vec3f sample_Gaussian(const CGLA::Vec3f& center, const CGLA::Vec3f& normal, const CGLA::Vec3f& tangent, double radius, double& r)
{
  static const double M_1_EXP = exp(-1.0);

  // Get random numbers
  r = mt_random();
  while(r < M_1_EXP)
    r = mt_random();
  r = radius*sqrt(-log(r));
  double phi = 2.0*M_PI*mt_random();

  // Calculate displacement by rotating tangent to new position
  CGLA::Quatf qrot;
  qrot.make_rot(phi, normal);
  CGLA::Vec3f displacement = qrot.apply_unit(tangent)*r;

  // Return sampled position
  return center + displacement;
}

inline CGLA::Vec3f sample_Gaussian(const CGLA::Vec3f& center, const CGLA::Vec3f& normal, const CGLA::Vec3f& tangent, double radius)
{
  double r;
  return sample_Gaussian(center, normal, tangent, radius, r);
}

inline CGLA::Vec3f sample_exp_plane(const CGLA::Vec3f& center, const CGLA::Vec3f& normal, const CGLA::Vec3f& tangent, double fall_off)
{
  // Get random numbers
  double r = -log(mt_random())/fall_off;
  double phi = 2.0*M_PI*mt_random();

  // Calculate displacement by rotating tangent to new position
  CGLA::Quatf qrot;
  qrot.make_rot(phi, normal);
  CGLA::Vec3f displacement = qrot.apply_unit(tangent)*r;

  // Return sampled position
  return center + displacement;
}

inline CGLA::Vec3f sample_Barycentric()
{
  // Get random numbers
  const float sqrt_xi1 = sqrt(mt_random());
  const float xi2 = mt_random();

  // Calculate Barycentric coordinates
  const float u = 1.0f - sqrt_xi1;
  const float v = (1.0f - xi2)*sqrt_xi1;
  const float w = xi2*sqrt_xi1;

  // Return Barycentric coordinates
  return CGLA::Vec3f(u, v, w);
}

inline CGLA::Vec3f sample_triangle(const CGLA::Vec3f& v0, const CGLA::Vec3f& v1, const CGLA::Vec3f& v2)
{
  CGLA::Vec3f uvw = sample_Barycentric();
  return v0*uvw[0] + v1*uvw[1] + v2*uvw[2];
}

#endif
