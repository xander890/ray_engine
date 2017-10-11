/*
 * Copyright (c) 2008 - 2013 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */
#ifndef OHELP
#define OHELP

namespace optix {

    __forceinline__ RT_HOSTDEVICE void operator/=(float3& a, const float3 s)
    {
        a.x /= s.x;
        a.y /= s.y;
        a.z /= s.z;
    }

    enum Axis {XAXIS, YAXIS, ZAXIS};

    __forceinline__ Matrix3x3 rotation_matrix3x3(Axis axis, float angle)
    {
        Matrix3x3 m;
        float c = cos(angle);
        float s = sin(angle);

        switch (axis)
        {
        case XAXIS:
            m.setRow(0, make_float3(1,0,0));
            m.setRow(1, make_float3(0,c,s));
            m.setRow(2, make_float3(0,-s,c));
            break;
        case YAXIS:
            m.setRow(0, make_float3(c,0,-s));
            m.setRow(1, make_float3(0,1,0));
            m.setRow(2, make_float3(s,0,c));
            break;
        case ZAXIS:
            m.setRow(0, make_float3(c,s,0));
            m.setRow(1, make_float3(-s,c,0));
            m.setRow(2, make_float3(0,0,1));
            break;
        }

        return m;
    }

    __forceinline__ Matrix3x3 scaling_matrix3x3(const float3& v)
    {
        Matrix3x3 m = Matrix3x3::identity();
        m.setRow(0, make_float3(v.x, 0, 0));
        m.setRow(1, make_float3(0, v.y, 0));
        m.setRow(2, make_float3(0, 0, v.z));
        return m;
    }

	__device__ __forceinline__ const float& get_channel(int C, const optix::float3 & elem)
	{
		return reinterpret_cast<const float*>(&elem)[C];
	}

	__device__ __forceinline__ float& get_channel(int C, optix::float3 & elem)
	{
		return reinterpret_cast<float*>(&elem)[C];
	}

}
#endif