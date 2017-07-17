#pragma once

#include<vector>
#include <algorithm>


struct Distribution1D
{
    Distribution1D(float *f, int n)
    {
        func = new float[n];
        cdf = new float[n+1];
        count = n;
        memcpy(func, f, n*sizeof(float));
        ComputeStep1dCDF(func, n, &funcInt, cdf);
        invFuncInt = 1.f / funcInt;
        invCount = 1.f / count;
    }

    void ComputeStep1dCDF(float *f, int nSteps, float *c, float *cdf) {
            // Compute integral of step function at $x_i$
            int i;
            cdf[0] = 0.;
            for (i = 1; i < nSteps+1; ++i)
                    cdf[i] = cdf[i-1] + f[i-1] / nSteps;
            // Transform step function integral into cdf
            *c = cdf[nSteps];
			
            for (i = 1; i < nSteps+1; ++i)
                    cdf[i] /= *c;
    }

    float Sample(float u, float * pdf)
    {
        // Find surrounding cdf segments
        float *ptr = std::lower_bound(cdf, cdf+count+1, u);
        int offset = (int) (ptr-cdf-1);
        // Return offset along current cdf segment
        u = (u - cdf[offset]) / (cdf[offset+1] - cdf[offset]);
        *pdf = func[offset] * invFuncInt;
        return offset + u;
    }

    float *func, *cdf;
    float funcInt, invFuncInt, invCount;
    int count;
};

