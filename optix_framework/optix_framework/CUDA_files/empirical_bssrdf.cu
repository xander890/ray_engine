#include "empirical_bssrdf_device.h"

RT_CALLABLE_PROGRAM optix::float3 eval_empirical_bssrdf(const BSSRDFGeometry geometry, const float recip_ior,
                                                            const MaterialDataCommon material)
{
    return eval_empbssrdf(geometry, recip_ior, material);
}

__global__ void stub()
{
    BSSRDFGeometry geometry; float recip_ior;MaterialDataCommon material;
    printf("%f", eval_empirical_bssrdf(geometry, recip_ior,material).x);
}