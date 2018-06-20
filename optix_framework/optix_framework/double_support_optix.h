#pragma once
#include "vector_functions.hpp"



namespace optix
{

	__forceinline__ RT_HOSTDEVICE double2 operator*(const double2& a, const double2& b)
	{
		return make_double2(a.x * b.x, a.y * b.y);
	}

	__forceinline__ RT_HOSTDEVICE double2 operator*(const double2& a, const float b)
	{
		return make_double2(a.x * b, a.y * b);
	}

	_fn double fmaxf(double a, double b)
	{
		return a > b ? a : b;
	}

	_fn double fminf(double a, double b)
	{
		return a < b ? a : b;
	}

	_fn double clamp(const double f, const double a, const double b)
	{
		return fmaxf(a, fminf(f, b));
	}

	__forceinline__ RT_HOSTDEVICE double3 make_double3c(const double x, const double y, const double z)
	{
		double3 t; t.x = x; t.y = y; t.z = z; return t;
	}

	__forceinline__ RT_HOSTDEVICE double3 make_double3c(int x, int y, int z)
	{
		return make_double3c(double(x), double(y), double(z));
	}

	__forceinline__ RT_HOSTDEVICE double3 make_double3(const double s)
	{
		return optix::make_double3c(s, s, s);
	}
	__forceinline__ RT_HOSTDEVICE double3 make_double3(const optix::double2& a)
	{
		return make_double3c(a.x, a.y, 0.0f);
	}
	__forceinline__ RT_HOSTDEVICE double3 make_double3(const int3& a)
	{
		return make_double3c(double(a.x), double(a.y), double(a.z));
	}
	__forceinline__ RT_HOSTDEVICE double3 make_double3(const float3& a)
	{
		return make_double3c(double(a.x), double(a.y), double(a.z));
	}
	__forceinline__ RT_HOSTDEVICE double3 make_double3(const uint3& a)
	{
		return make_double3c(double(a.x), double(a.y), double(a.z));
	}
	/** @} */

	/** negate */
	__forceinline__ RT_HOSTDEVICE double3 operator-(const double3& a)
	{
		return make_double3(-a.x, -a.y, -a.z);
	}

	/** min
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 fminf(const double3& a, const double3& b)
	{
		return make_double3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
	}
	__forceinline__ RT_HOSTDEVICE double fminf(const double3& a)
	{
		return fminf(fminf(a.x, a.y), a.z);
	}
	/** @} */

	/** max
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 fmaxf(const double3& a, const double3& b)
	{
		return make_double3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
	}
	__forceinline__ RT_HOSTDEVICE double fmaxf(const double3& a)
	{
		return fmaxf(fmaxf(a.x, a.y), a.z);
	}
	/** @} */

	/** add
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 operator+(const double3& a, const double3& b)
	{
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator+(const double3& a, const double b)
	{
		return make_double3(a.x + b, a.y + b, a.z + b);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator+(const double a, const double3& b)
	{
		return make_double3(a + b.x, a + b.y, a + b.z);
	}
	__forceinline__ RT_HOSTDEVICE void operator+=(double3& a, const double3& b)
	{
		a.x += b.x; a.y += b.y; a.z += b.z;
	}
	/** @} */

	/** subtract
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 operator-(const double3& a, const double3& b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator-(const double3& a, const double b)
	{
		return make_double3(a.x - b, a.y - b, a.z - b);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator-(const double a, const double3& b)
	{
		return make_double3(a - b.x, a - b.y, a - b.z);
	}
	__forceinline__ RT_HOSTDEVICE void operator-=(double3& a, const double3& b)
	{
		a.x -= b.x; a.y -= b.y; a.z -= b.z;
	}
	/** @} */

	/** multiply
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 operator*(const double3& a, const double3& b)
	{
		return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator*(const double3& a, const double s)
	{
		return make_double3(a.x * s, a.y * s, a.z * s);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator*(const double s, const double3& a)
	{
		return make_double3(a.x * s, a.y * s, a.z * s);
	}
	__forceinline__ RT_HOSTDEVICE void operator*=(double3& a, const double3& s)
	{
		a.x *= s.x; a.y *= s.y; a.z *= s.z;
	}
	__forceinline__ RT_HOSTDEVICE void operator*=(double3& a, const double s)
	{
		a.x *= s; a.y *= s; a.z *= s;
	}
	/** @} */

	/** divide
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 operator/(const double3& a, const double3& b)
	{
		return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
	}
	__forceinline__ RT_HOSTDEVICE double3 operator/(const double3& a, const double s)
	{
		double inv = 1.0f / s;
		return a * inv;
	}
	__forceinline__ RT_HOSTDEVICE double3 operator/(const double s, const double3& a)
	{
		return make_double3(s / a.x, s / a.y, s / a.z);
	}
	__forceinline__ RT_HOSTDEVICE void operator/=(double3& a, const double s)
	{
		double inv = 1.0f / s;
		a *= inv;
	}
	/** @} */

	/** lerp */
	__forceinline__ RT_HOSTDEVICE double3 lerp(const double3& a, const double3& b, const double t)
	{
		return a + t*(b - a);
	}

	/** bilerp */
	__forceinline__ RT_HOSTDEVICE double3 bilerp(const double3& x00, const double3& x10, const double3& x01, const double3& x11,
		const double u, const double v)
	{
		return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
	}

	/** clamp
	* @{
	*/
	__forceinline__ RT_HOSTDEVICE double3 clamp(const double3& v, const double a, const double b)
	{
		return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
	}

	__forceinline__ RT_HOSTDEVICE double3 clamp(const double3& v, const double3& a, const double3& b)
	{
		return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
	}
	/** @} */

	/** dot product */
	__forceinline__ RT_HOSTDEVICE double dot(const double3& a, const double3& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	/** cross product */
	__forceinline__ RT_HOSTDEVICE double3 cross(const double3& a, const double3& b)
	{
		return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}

	/** length */
	__forceinline__ RT_HOSTDEVICE double length(const double3& v)
	{
		return sqrtf(dot(v, v));
	}

	/** normalize */
	__forceinline__ RT_HOSTDEVICE double3 normalize(const double3& v)
	{
		double invLen = 1.0f / sqrtf(dot(v, v));
		return v * invLen;
	}

	/** floor */
	__forceinline__ RT_HOSTDEVICE double3 floor(const double3& v)
	{
		return make_double3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
	}
}