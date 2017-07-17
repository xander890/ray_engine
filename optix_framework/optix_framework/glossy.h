#ifndef glossy_h__
#define glossy_h__
#include "shader.h"

class GlossyShader : public Shader
{
public:
    virtual ~GlossyShader() = default;
    GlossyShader() : Shader() {}

    void initialize_mesh(Mesh & object) override;
    void pre_trace_mesh(Mesh & object) override;

};


#endif // glossy_h__
