#pragma once

template<typename T, int N>
class vec
{
private:
	T data[N];
public:
	T& operator[](int index)
	{
		return data[index];
	}
};

using Vec2f = vec<float, 2>;
using Vec3f = vec<float, 3>;
using Vec4f = vec<float, 4>;

using Vec2i = vec<int, 2>;
using Vec3i = vec<int, 3>;
using Vec4i = vec<int, 4>;