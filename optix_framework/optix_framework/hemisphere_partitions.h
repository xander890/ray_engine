#pragma once

#include <vector>
#include "host_device_common.h"
#include "logger.h"
#include <cmath>

float isqrt(float c)
{
    return roundf(sqrtf(c));
}
inline int get_elements_ring2(int ring, float p)
{
    if(ring == 0)
        return 1;

    int n_next = 1;
    for(int i = 0; i < ring; i++)
    {
        float nn = float(n_next);
        n_next = int(isqrt(p + nn * nn + 2.0f * nn * isqrt(p)));
    }
    return n_next * n_next;
}

inline int get_elements_ring(int ring, float p)
{
    int k = 1;

    for(int i = 0; i < ring; i++)
    {
        float l = sqrtf(float(k)) + sqrtf(p);
        k = int(roundf(l*l));
    }
    return k;
}

inline void get_hemisphere_subdivisions(const int desired_elements, std::vector<float>& rad, std::vector<int>& ks)
{
    int N = 0;
    const float p = M_PIf;

    int k = 1;
    while(k < desired_elements)
    {
        float l = sqrtf(float(k)) + sqrtf(p);
        k = int(roundf(l*l));
        N++;
    }

    Logger::info << "Elements: desired " << desired_elements << " given: " << k << std::endl;

    std::vector<float> radii(N+1, 0.0f);
    std::vector<int> elements(N+1, 0);

    elements[N] = get_elements_ring(N, p);
    radii[N] = 1.0f;

    for(int i = N-1; i >= 0; i--)
    {
        elements[i] = get_elements_ring(i, p);
        float k_i_minus = float(elements[i]);
        float k_i = float(elements[i + 1]);
        float r_prev = radii[i + 1];
        radii[i] = r_prev * sqrtf(k_i_minus / k_i);

/*        int diff = k_i - k_i_minus;
        float r_minus = r_p;
        float r_plus = rad.back();
        float a = M_PIf / diff * (r_plus*r_plus - r_minus*r_minus) / ((r_plus - r_minus)*(r_plus - r_minus));
*/
    }

    float theta = M_PIf / 2.0f;
    rad.clear();
    ks.clear();

    rad.push_back(1.0f);
    ks.push_back(k);
    std::cout << ks.back() << " " << rad.back() << std::endl;

    while(true)
    {
        int k_p = ks.back();
        float r_p = rad.back();
        theta = theta - 2.0f * sinf(theta / 2.0f) * sqrtf(M_PIf / float(k_p));

        float r_new = sqrtf(2.0f) * sinf(theta / 2.0f);
        int k_new = int(roundf(float(k_p) * r_new*r_new / (r_p*r_p)));

        if(k_new <= 0 || theta < 0.0f)
            break;
        ks.push_back(k_new);
        rad.push_back(r_new);

        int diff = ks[ks.size() - 1] - ks[ks.size() - 2];
        float r_minus = r_p;
        float r_plus = rad.back();
        float a = M_PIf / diff * (r_plus*r_plus - r_minus*r_minus) / ((r_plus - r_minus)*(r_plus - r_minus));
        std::cout << ks.back() << " " << a <<  std::endl;
    }

}