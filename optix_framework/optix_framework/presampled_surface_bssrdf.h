#ifndef PRESAMPLED_H
#define PRESAMPLED_H

#include <optix_world.h>
#include "rendering_method.h"

using namespace optix;

const unsigned int SAMPLES_FRAME = 100;


class PresampledSurfaceBssrdf : public Shader 
{
public:
    PresampledSurfaceBssrdf() : Shader(), entry_point(0) { }

    void initialize_shader(optix::Context, int illum) override;
    void initialize_mesh(Mesh& object) override;
    void pre_trace_mesh(Mesh& object) override;

private:
    int entry_point;
    Buffer m_samples;
};



#endif // PRESAMPLED_H
