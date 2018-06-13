#include <device_common_data.h>
#include <math_helpers.h>
#include <random.h>
#include <optical_helper.h>
#include <optix_helpers.h>
#include <ray_trace_helpers.h>
#include <scattering_properties.h>
#include <sampling_helpers.h>
#include "material.h"
#include "bssrdf_properties.h"
#include "empirical_bssrdf_common.h"

// HyperNetwork parameters
rtDeclareVariable(int, hypernetwork_num_layers, , );

rtBuffer<float> hypernetwork_layer1_weights;
rtDeclareVariable(int, hypernetwork_layer1_in_size, ,);
rtDeclareVariable(int, hypernetwork_layer1_out_size, ,);
rtBuffer<float> hypernetwork_layer1_biases;

rtBuffer<float> hypernetwork_layer2_weights;
rtDeclareVariable(int, hypernetwork_layer2_in_size, ,);
rtDeclareVariable(int, hypernetwork_layer2_out_size, ,);
rtBuffer<float> hypernetwork_layer2_biases;

rtBuffer<float> hypernetwork_layer3_weights;
rtDeclareVariable(int, hypernetwork_layer3_in_size, ,);
rtDeclareVariable(int, hypernetwork_layer3_out_size, ,);
rtBuffer<float> hypernetwork_layer3_biases;

rtDeclareVariable(int, integral_texture, ,);

_fn float get_interpolated_integral(const float albedo, const float g, const float theta_i)
{
    const float albedo_normalized = albedo;
    const float g_normalized = g; // FIXME if using negative gs with (g + 1) / 2
    const float theta_i_normalized = theta_i * 2 / M_PIf;
    optix::float4 value = optix::rtTex3D<optix::float4>(integral_texture, albedo_normalized, g_normalized, theta_i_normalized);
    return value.x;
}

_fn void mat_mul(int rows, int cols, int inner, float *a, float *b, float *c)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            c[i * cols + j] = 0.0;
            for (int k = 0; k < inner; ++k)
            {   
                c[i * cols + j] += a[i * inner + k] * b[k * cols + j];
            }
        }
    }
}

_fn void set_cdfnetwork_parameters(
        int layer_index,     // CDF network layer index
        int num_layers,      // CDF network number of layers
        float *params,       // Hypernetwork output parameters 
        int in_size,         // CDF network layer dimensions
        int out_size,
        optix::Matrix4x4 &W, // CDF network layer weights
        float *b,            // CDF network layer biases
        float *c)            // CDF network layer additional learnable weights
{
    // Stride begins from the front
    int initial_stride = hypernetwork_layer3_out_size / num_layers * (layer_index);

    // Weights 
    for (int i = 0; i < in_size; ++i)
    {
        for (int j = 0; j < out_size; ++j)
        {
            int index = i * out_size + j;
            W[index] = params[initial_stride + index];
        }
    }

    // Biases and additional learnable weights
    int len_W = in_size * out_size;
    for (int i = 0; i < out_size; ++i)
    {
        b[i] = params[initial_stride + len_W + i];
        c[i] = params[initial_stride + len_W + out_size + i];
    }
}

_fn void get_cdfnetwork_output(float *input, float *params, float *output)
{
    // CDF network dimensions
    int D_in = 4;
    int H = 4;
    int D_out = 4;
    int num_layers = 3;

    // Set CDF network parameters
    optix::Matrix4x4 cdfnetwork_W[3];
    float cdfnetwork_b[3][4];
    float cdfnetwork_c[3][4];

    set_cdfnetwork_parameters(0, num_layers, params, D_in, H, cdfnetwork_W[0], &cdfnetwork_b[0][0], &cdfnetwork_c[0][0]);
    set_cdfnetwork_parameters(1, num_layers, params, H, H, cdfnetwork_W[1], &cdfnetwork_b[1][0], &cdfnetwork_c[1][0]);
    set_cdfnetwork_parameters(2, num_layers, params, H, D_out, cdfnetwork_W[2], &cdfnetwork_b[2][0], &cdfnetwork_c[2][0]);

    // Layer 1: Expression Layer
    //     input: 1 x D_in (dimensions)
    //     output: 1 x D_in

    //     Map from (0, 1) to (-inf, inf)
    //     s2 = atanh(2 * input - 1)
    float s2[4];
    for (int i = 0; i < D_in; ++i)
    {
        s2[i] = atanh(2.0 * input[i] - 1.0);
    }

    // Layer 2: Linear Layer
    //     input: 1 x D_in (dimensions)
    //     weights: D_in x H
    //     biases: 1 x H
    //     c: 1 x H
    //     output: 1 x H

    //     Hyperbolic tangent nonlinear activation
    //     s3 = asinh(c1 + sinh(b1 + s2.dot(W1)))
    float s3[4];
    mat_mul(1, H, D_in, &s2[0], &cdfnetwork_W[0][0], &s3[0]);

    for (int i = 0; i < H; ++i)
    {
        s3[i] = asinh(cdfnetwork_c[0][i] + sinh(cdfnetwork_b[0][i] + s3[i]));
    }

    // Layer 3: Linear Layer
    //     input: 1 x H (dimensions)
    //     weights: H x H
    //     biases: 1 x H
    //     c: 1 x H
    //     output: 1 x H

    //     Hyperbolic tangent nonlinear activation
    //     s4 = asinh(c2 + sinh(b2 + s3.dot(W2)))
    float s4[4];
    mat_mul(1, H, H, &s3[0], &cdfnetwork_W[1][0], &s4[0]);

    for (int i = 0; i < H; ++i)
    {
        s4[i] = asinh(cdfnetwork_c[1][i] + sinh(cdfnetwork_b[1][i] + s4[i]));
    }

    // Layer 4: Linear Layer
    //     input: 1 x H (dimensions)
    //     weights: H x D_out
    //     biases: 1 x D_out
    //     c: 1 x D_out
    //     output: 1 x D_out

    //     Hyperbolic tangent nonlinear activation
    //     s5 = asinh(c3 + sinh(b3 + s4.dot(W3)))
    float s5[4];
    mat_mul(1, D_out, H, &s4[0], &cdfnetwork_W[2][0], &s5[0]);

    for (int i = 0; i < D_out; ++i)
    {
        s5[i] = asinh(cdfnetwork_c[2][i] + sinh(cdfnetwork_b[2][i] + s5[i]));
    }

    // Layer 5: Expression Layer
    //     input: 1 x D_out (dimensions)
    //     output: 1 x D_out

    //     Map from (-inf, inf) to (0, 1)
    //     output = tanh(s5) * 0.5 + 0.5
    for (int i = 0; i < D_out; ++i)
    {
        output[i] = tanh(s5[i]) * 0.5 + 0.5;
    }
}

_fn void get_icdfnetwork_output(float *input, float *params, float *output)
{
    // CDF network dimensions
    int D_in = 4;
    int H = 4;
    int D_out = 4;
    int num_layers = 3;

    // Set CDF network parameters
    optix::Matrix4x4 cdfnetwork_W[3];
    float cdfnetwork_b[3][4];
    float cdfnetwork_c[3][4];

    set_cdfnetwork_parameters(0, num_layers, params, D_in, H, cdfnetwork_W[2], &cdfnetwork_b[2][0], &cdfnetwork_c[2][0]);
    set_cdfnetwork_parameters(1, num_layers, params, H, H, cdfnetwork_W[1], &cdfnetwork_b[1][0], &cdfnetwork_c[1][0]);
    set_cdfnetwork_parameters(2, num_layers, params, H, D_out, cdfnetwork_W[0], &cdfnetwork_b[0][0], &cdfnetwork_c[0][0]);

    // Layer 1: Expression Layer
    //     input: 1 x D_in (dimensions)
    //     output: 1 x D_in

    //     Map from (0, 1) to (-inf, inf)
    //     s2 = atanh(2 * input - 1)
    float s2[4];
    for (int i = 0; i < D_in; ++i)
    {
        s2[i] = atanh(2.0 * input[i] - 1.0);
    }

    // Layer 2: Linear Layer
    //     input: 1 x D_in (dimensions)
    //     weights: D_in x H
    //     biases: 1 x H
    //     c: 1 x H
    //     output: 1 x H

    //     Hyperbolic tangent nonlinear activation
    //     s3 = asinh(-c3 + sinh(s2))
    //     s3 = s3 - b3
    //     s3 = s3.dot(inv(W3))
    float s3[4];
    for (int i = 0; i < D_in; ++i)
    {
        output[i] = asinh(-cdfnetwork_c[0][i] + sinh(s2[i]));
        output[i] -= cdfnetwork_b[0][i];
    }

    optix::Matrix4x4 temp = cdfnetwork_W[0].inverse();
    mat_mul(1, H, D_in, &output[0], &temp[0], &s3[0]);

    // Layer 3: Linear Layer
    //     input: 1 x H (dimensions)
    //     weights: H x H
    //     biases: 1 x H
    //     c: 1 x H
    //     output: 1 x H

    //     Hyperbolic tangent nonlinear activation
    //     s4 = asinh(-c2 + sinh(s3))
    //     s4 = s4 - b2
    //     s4 = s4.dot(inv(W2))
    float s4[4];
    for (int i = 0; i < H; ++i)
    {
        output[i] = asinh(-cdfnetwork_c[1][i] + sinh(s3[i]));
        output[i] -= cdfnetwork_b[1][i];
    }

    temp = cdfnetwork_W[1].inverse();
    mat_mul(1, H, H, &output[0], &temp[0], &s4[0]);

    // Layer 3: Linear Layer
    //     input: 1 x H (dimensions)
    //     weights: H x D_out
    //     biases: 1 x D_out 
    //     c: 1 x D_out
    //     output: 1 x D_out

    //     Hyperbolic tangent nonlinear activation
    //     s5 = asinh(-c1 + sinh(s4))
    //     s5 = s5 - b1
    //     s5 = s5.dot(inv(W1))
    float s5[4];
    for (int i = 0; i < H; ++i)
    {
        output[i] = asinh(-cdfnetwork_c[2][i] + sinh(s4[i]));
        output[i] -= cdfnetwork_b[2][i];
    }

    temp = cdfnetwork_W[2].inverse();
    mat_mul(1, D_out, H, &output[0], &temp[0], &s5[0]);

    // Layer 5: Expression Layer
    //     input: 1 x D_out (dimensions)
    //     output: 1 x D_out

    //     Map from (-inf, inf) to (0, 1)
    //     output = tanh(s5) * 0.5 + 0.5
    for (int i = 0; i < D_out; ++i)
    {
        output[i] = tanh(s5[i]) * 0.5 + 0.5;
    }
}

_fn void get_hypernetwork_output(float *input, float *output)
{
    // Layer 1: DenseLayer
    //    input: 1 x 4 (dimensions)
    //    weights: hypernetwork_layer1_in_size x hypernetwork_layer1_out_size
    //    biases: 1 x hypernetwork_layer1_out_size
    //    output: 1 x hypernetwork_layer1_out_size

    //    s2 = input.dot(weights) + biases
    //    s2 = sigmoid(s2)
    float s2[6];
    mat_mul(1, hypernetwork_layer1_out_size, hypernetwork_layer1_in_size, input, &hypernetwork_layer1_weights[0], &s2[0]);

    for (int i = 0; i < hypernetwork_layer1_out_size; ++i)
    {   
        s2[i] += hypernetwork_layer1_biases[i];
        s2[i] = 1.0 / (1.0 + exp(-s2[i]));
    }

    // Layer 2: DenseLayer
    //    input: 1 x 6 (dimensions)
    //    weights: hypernetwork_layer2_in_size x hypernetwork_layer2_out_size
    //    biases: 1 x hypernetwork_layer2_out_size
    //    output: 1 x hypernetwork_layer2_out_size

    //    s3 = s2.dot(weights) + biases
    //    s3 = sigmoid(s3)
    float s3[6];
    mat_mul(1, hypernetwork_layer2_out_size, hypernetwork_layer2_in_size, &s2[0], &hypernetwork_layer2_weights[0], &s3[0]);
    
    for (int i = 0; i < hypernetwork_layer2_out_size; ++i)
    {   
        s3[i] += hypernetwork_layer2_biases[i];
        s3[i] = 1.0 / (1.0 + exp(-s3[i]));
    }

    // Layer 3: DenseLayer
    //    input: 1 x 6 (dimensions)
    //    weights: hypernetwork_layer3_in_size x hypernetwork_layer3_out_size
    //    biases: 1 x hypernetwork_layer3_out_size
    //    output: 1 x hypernetwork_layer3_out_size

    //    output = s3.dot(weights) + biases
    mat_mul(1, hypernetwork_layer3_out_size, hypernetwork_layer3_in_size, &s3[0], &hypernetwork_layer3_weights[0], output);
    
    for (int i = 0; i < hypernetwork_layer3_out_size; ++i)
    {
        output[i] += hypernetwork_layer3_biases[i];
    }
}

_fn float map_interval(float point, optix::float2 interval_from, optix::float2 interval_to)
{
    float map_01 = (point - interval_from.x)/(interval_from.y - interval_from.x);
    return interval_to.x + map_01 * (interval_to.y - interval_to.x);
}

_fn void sample_neural_network(
        const optix::float3 & xo,          // The points hit by the camera ray.
        const optix::float3 & no,          // The normal at the point.
        const optix::float3 & wo,          // The incoming ray direction.
        const MaterialDataCommon & material,  // Material properties.
        const int colorband,        // Specific colorband frequency.
        TEASampler * sampler,       // A rng.
        optix::float3 & x_tangent,                // The candidate point
        float & integration_factor, // An factor that will be multiplied into the final result. For inverse pdfs. 
        optix::float3 & proposed_wi) {
    // Gathering scattering parameters.
    // For now only red channel.
    float albedo = optix::get_channel(colorband, material.scattering_properties.albedo);
    float extinction = optix::get_channel(colorband, material.scattering_properties.extinction);
    float eta = dot(material.index_of_refraction, optix::make_float3(1)) / 3.0f;
    float g = optix::get_channel(colorband, material.scattering_properties.meancosine);

    float cos_theta_i = dot(wo, no);
    float theta_i = acosf(cos_theta_i);

    // Sampling random inputs for NN.
    float x1 = sampler->next1D();
    float x2 = sampler->next1D();
    float x3 = sampler->next1D();
    float x4 = sampler->next1D();

    // FIXME code should be available about colorband from now on.
    // Get hypernetwork output for the specified optical parameters. A set of
    // optical parameters are passed to the hypernetwork which will generate
    // parameters for the iCDF network.
    // The hypernetwork output dimensions should match 
    // hypernetwork_layer3_out_size. Preallocated arrays are used here instead
    // of dynamic memory allocation.
    float hypernetwork_input[4] = {eta, g, albedo, theta_i};
    float hypernetwork_output[72];
    get_hypernetwork_output(&hypernetwork_input[0], &hypernetwork_output[0]);

    // Get iCDF network output. We pass a set of random numbers through the
    // iCDF network to get the geometry parameters. We then use these geometry
    // parameters to get an outgoing direction.
    float icdfnetwork_input[4] = {x1, x2, x3, x4};
    float icdfnetwork_output[4];
    get_icdfnetwork_output(&icdfnetwork_input[0], &hypernetwork_output[0], &icdfnetwork_output[0]);

    // Get CDF network output to verify iCDF network
    float cdfnetwork_output[4];
    get_cdfnetwork_output(&icdfnetwork_output[0], &hypernetwork_output[0], &cdfnetwork_output[0]);

#if 0
    for (int i = 0; i < hypernetwork_layer3_out_size; ++i)
    {
        optix_print("%f ", hypernetwork_output[i]);
    }
    optix_print("\n");
#endif
    optix_print("icdfnetwork_input: %f %f %f %f\n", x1, x2, x3, x4);
    optix_print("icdfnetwork_output: ");
    for (int i = 0; i < 4; ++i) {
        optix_print("%f ", icdfnetwork_output[i]);
    }
    optix_print("\n");
    optix_print("cdfnetwork_output: ");
    for (int i = 0; i < 4; ++i) {
        optix_print("%f ", cdfnetwork_output[i]);
    }
    optix_print("\n");

    // Also here sample normalization constant
    float bssrdf_integral = get_interpolated_integral(albedo, g, theta_i);
    integration_factor *= bssrdf_integral;
    // Multiplying over extinction as in the empbssrdf paper
    integration_factor *= extinction / extinction;

    float r = map_interval(icdfnetwork_output[0], optix::make_float2(0, 1), optix::make_float2(0.01f, 10.0f));
    float theta_s = map_interval(icdfnetwork_output[1], optix::make_float2(0, 1), optix::make_float2(0.0f, M_PIf));
    float theta_o = map_interval(icdfnetwork_output[2], optix::make_float2(0, 1), optix::make_float2(0.0f, M_PIf / 2));
    float phi_o = map_interval(icdfnetwork_output[3], optix::make_float2(0, 1), optix::make_float2(0.0f, 2 * M_PIf));

    // The radius is expressed in mean free paths, so we renormalize it.
    r /= extinction;

    // Pick a random side for theta_s
    float zeta = sampler->next1D();
    theta_s *= (zeta > 0.5f) ? -1 : 1;
    integration_factor *= 2;

    // Note that the tangent vector has to be aligned to wo in order to get a consistent framae for theta_s.
    BSSRDFGeometry geometry_exit;

#define TREAT_AS_INPUT_GEOMETRY
#ifdef TREAT_AS_INPUT_GEOMETRY
    empirical_bssrdf_build_geometry(xo, wo, no, theta_i, r, theta_s, theta_o, phi_o, geometry_exit);

    x_tangent = geometry_exit.xo;
    proposed_wi = geometry_exit.wo;
#else
    empirical_bssrdf_build_geometry_from_exit(xo, wo, no, theta_i, r, theta_s, theta_o, phi_o, geometry_exit);
    x_tangent = geometry_exit.xi;
    proposed_wi = geometry_exit.wi;
#endif
}
