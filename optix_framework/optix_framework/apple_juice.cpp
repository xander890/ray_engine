#include "apple_juice.h"
#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include "optprops/cdf_bsearch.h"
#include "optprops/LorenzMie.h"

using namespace std;
using namespace LorenzMie;

namespace
{
#ifndef M_PI
  const double M_PI = 3.14159265358979323846;
#endif

  // Data from Hale and Querry [1973] and Pope and Fry [1997]
  complex<double> refrac_water[] = { complex<double>(1.341, 3.393e-10),
                                     complex<double>(1.339, 2.110e-10),
                                     complex<double>(1.338, 1.617e-10),
                                     complex<double>(1.337, 3.302e-10),
                                     complex<double>(1.336, 4.309e-10),
                                     complex<double>(1.335, 8.117e-10),
                                     complex<double>(1.334, 1.742e-9),
                                     complex<double>(1.333, 2.473e-9),
                                     complex<double>(1.333, 3.532e-9),
                                     complex<double>(1.332, 1.062e-8),
                                     complex<double>(1.332, 1.410e-8),
                                     complex<double>(1.331, 1.759e-8),
                                     complex<double>(1.331, 2.406e-8),
                                     complex<double>(1.331, 3.476e-8),
                                     complex<double>(1.330, 8.591e-8),
                                     complex<double>(1.330, 1.474e-7),
                                     complex<double>(1.330, 1.486e-7)  };

  // Absorbance of browned clarified apple juice [Beveridge et al. 1986].
  // Wavelength range: 400 to 700 nanometers (steps of 25 nanometers).
  // Presumably measured in juice of 2 cm thickness.
  // The first array range:
  //   0 =  96 hours storage at 37 degrees Celsius
  //   1 = 228 hours storage at 37 degrees Celsius (peeled and cored)
  //   2 = 228 hours storage at 37 degrees Celsius
  //   3 = 648 hours storage at 37 degrees Celsius
  double apple_juice_abs[4][13] = { { 0.46, 0.19, 0.13, 0.125, 0.13, 0.125, 0.10,  0.075, 0.054, 0.040, 0.033, 0.029, 0.025 },
                                    { 0.30, 0.25, 0.24, 0.22,  0.18, 0.13,  0.096, 0.075, 0.058, 0.050, 0.048, 0.047, 0.047 },
                                    { 0.64, 0.39, 0.34, 0.30,  0.24, 0.18,  0.12,  0.083, 0.058, 0.046, 0.038, 0.033, 0.029 },
                                    { 0.93, 0.78, 0.78, 0.67,  0.47, 0.31,  0.20,  0.14,  0.10,  0.075, 0.058, 0.046, 0.042 } };

  // Apple flesh absorption coefficients from Saeys et al. [2008]. The first array range:
  //   0 = wavelength (nanometers)
  //   1 = absorption coefficient (per millimeter)
  double apple_abs[2][17] = { { 350.0, 355.0, 356.0, 364.0, 365.0, 382.0, 450.0, 455.0, 490.0, 
                                495.0, 500.0, 505.0, 520.0, 525.0, 632.0, 635.0, 930.0 },
                              { 0.72, 0.68, 0.58, 0.53, 0.47, 0.32, 0.32, 0.26, 0.26,
                                0.21, 0.21, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05 } };

  // Absorption coefficients (per centimeter) of bruised Golden Delicious apples [Lu et al. 2010].
  // Wavelength range: 500 to 800 nanometers (steps of 5 nanometers).
  // The first array range:
  //   0 = spectrum 2 days after bruising
  //   1 = spectrum 9 days after bruising
  double bruised_apple_abs[2][61] = { { 1.05,   1.06,   1.025,  0.98,   0.935,  0.86,   0.79,   0.72,   0.665, 
                                        0.62,   0.58,   0.55,   0.51,   0.48,   0.45,   0.42,   0.39,   0.365, 
                                        0.345,  0.33,   0.318,  0.31,   0.30,   0.292,  0.287,  0.284,  0.28, 
                                        0.28,   0.28,   0.28,   0.284,  0.30,   0.318,  0.332,  0.335,  0.332, 
                                        0.30,   0.25,   0.20,   0.15,   0.11,   0.077,  0.058,  0.044,  0.038, 
                                        0.036,  0.034,  0.0325, 0.0308, 0.0308, 0.0299, 0.0291, 0.0282, 0.0274, 
                                        0.0274, 0.0256, 0.0239, 0.0205, 0.0179, 0.0154, 0.0137 },
                                      { 1.33,  1.39,  1.365, 1.35,  1.31,  1.24,  1.18,  1.12,  1.07,
                                        1.024, 0.986, 0.96,  0.935, 0.912, 0.888, 0.873, 0.838, 0.818,
                                        0.800, 0.776, 0.753, 0.740, 0.729, 0.705, 0.700, 0.694, 0.682,
                                        0.671, 0.666, 0.679, 0.688, 0.694, 0.714, 0.728, 0.731, 0.706,
                                        0.674, 0.616, 0.529, 0.456, 0.384, 0.341, 0.311, 0.288, 0.265,
                                        0.247, 0.241, 0.231, 0.221, 0.216, 0.212, 0.205, 0.200, 0.195, 
                                        0.193, 0.182, 0.171, 0.166, 0.161, 0.156, 0.153 } };

  // Number density distribution of particles in the fine apple particle cloud.
  // The size distribution was found by manually fitting normal and log-normal
  // distributions to a measured curve [Beveridge 2002, Zimmer et al. 1994b].
  ParticleDistrib fine_cloud(double vol_frac)
  {
    ParticleDistrib distrib;
    distrib.r_min = -1.0;
    distrib.r_max = 2.0;
    distrib.dr = 3.0/39.0;
    distrib.N.resize(40);
    for(unsigned int i = 0; i < distrib.N.size(); ++i)
    {
      double log10_D = -1.0 + i*distrib.dr;
      double exponent = (log10_D + 0.32 + log(log10_D + 1.32))/0.464470;
      distrib.N[i] = vol_frac*0.133*exp(-0.5*exponent*exponent);
      double r = pow(10.0, log10_D)*0.5e-6;
      double dr = (pow(10.0, log10_D + 0.5*distrib.dr) - pow(10.0, log10_D - 0.5*distrib.dr))*0.5e-6;
      distrib.N[i] /= 4.0/3.0*M_PI*r*r*r*dr;
    }
    return distrib;
  }

  // Number density distribution of particles in the coarse apple particle cloud.
  // The size distribution was found by manually fitting a normal distribution
  // to a measured curve [Beveridge 2002, Zimmer et al. 1994b].
  ParticleDistrib coarse_cloud(double vol_frac)
  {
    ParticleDistrib distrib;
    distrib.r_min = -1.0;
    distrib.r_max = 2.0;
    distrib.dr = 3.0/39.0;
    distrib.N.resize(40);
    double total = 0.0;
    for(unsigned int i = 0; i < distrib.N.size(); ++i)
    {
      double log10_D = -1.0 + i*distrib.dr;
      double exponent = (log10_D - 1.147)/0.284384;
      distrib.N[i] = vol_frac*0.108*exp(-0.5*exponent*exponent);
      double r = pow(10.0, log10_D)*0.5e-6;
      double dr = (pow(10.0, log10_D + 0.5*distrib.dr) - pow(10.0, log10_D - 0.5*distrib.dr))*0.5e-6;
      distrib.N[i] /= 4.0/3.0*M_PI*r*r*r*dr;
    }
    return distrib;
  }

  template<class T> void init_spectrum(Color<T>& c, unsigned int no_of_samples)
  {
    c.resize(no_of_samples);
    c.wavelength = 375.0;
    c.step_size = 25.0;
  }

  // Modified optical_props function for integrating over log10-transformed
  // particle size distribution.
  void optical_props_log10(ParticleDistrib* p, 
                           double wavelength, 
                           const complex<double>& host_refrac,
                           const complex<double>* particle_refrac)
  {
    if(!p)
    {
      cerr << "Error: Particle distribution p is a null pointer.";
      return;
    }
    p->ext = 0.0;
    p->sca = 0.0;
    p->abs = 0.0;
    p->g = 0.0;
    p->ior = 0.0;
    for(unsigned int i = 0; i < p->N.size(); ++i)
    {
      double log10_D = p->r_min + i*p->dr;
      double r = pow(10.0, log10_D)*0.5e-6;
      double dr = (pow(10.0, log10_D + 0.5*p->dr) - pow(10.0, log10_D - 0.5*p->dr))*0.5e-6;
      double C_t, C_s, C_a, g, ior;
      particle_props(C_t, C_s, C_a, g, ior, 
                     r, wavelength,
                     host_refrac,
                     particle_refrac ? *particle_refrac : p->refrac_idx);
      
      double sigma_s = C_s*p->N[i]*dr;
      p->ext += C_t*p->N[i]*dr;
      p->sca += sigma_s;
      p->abs += C_a*p->N[i]*dr;
      p->g += g*sigma_s;
      p->ior += ior*p->N[i]*dr;
    }
    if(p->sca > 0.0)
      p->g /= p->sca;
  }
}

Medium apple_juice(double C)
{
  const double apple_density = 1.25;   // 1.2-1.25 [Beveridge 2002, Mensah-Wilson et al. 2000], 1.3 [Genovese et al. 1997]
  const double juice_density = 1.053;  // [Benitez et al. 2007], more data is available [Constenla et al. 1989, Genovese and Lozano 2006]

  // Apple particle size distribution is bimodal after centrifugation [Beveridge 2002, Genovese and Lozano 2006, Benitez et al. 2007]
  double fine_to_coarse = 117.0/(117.0 + 198.0); // Press juice [Beveridge 2002, Zimmer et al. 1994a]
  //double fine_to_coarse = 537.0/(537.0 + 594.0); // Decanter juice [Beveridge 2002, Zimmer et al. 1994a]

  C *= 1.0e-3; // g/L -> g/mL
  ParticleDistrib fine = fine_cloud(C*fine_to_coarse/apple_density);
  ParticleDistrib coarse = coarse_cloud(C*(1.0 - fine_to_coarse)/apple_density);
  ParticleDistrib apple;
  apple.refrac_idx = complex<double>(1.487, 0.0); // real part from Benitez et al. [2007]
  apple.r_min = fine.r_min;
  apple.r_max = coarse.r_max;
  apple.dr = fine.dr;
  apple.N.resize(std::max(fine.N.size(), coarse.N.size()));
  for(unsigned int i = 0; i < std::min(fine.N.size(), coarse.N.size()); ++i)
    apple.N[i] = fine.N[i] + coarse.N[i];

  const int no_of_samples = 17;
  Medium m;
  complex<double> refrac_host[no_of_samples];
  complex<double> refrac_apple[no_of_samples];
  Color< complex<double> >& ior = m.get_ior(spectrum);
  Color<double>& extinct = m.get_extinction(spectrum);
  Color<double>& scatter = m.get_scattering(spectrum);
  Color<double>& absorp = m.get_absorption(spectrum);
  Color<double>& asymmetry = m.get_asymmetry(spectrum);

  // Initialize medium
  init_spectrum(ior, no_of_samples);
  init_spectrum(extinct, no_of_samples);
  init_spectrum(scatter, no_of_samples);
  init_spectrum(absorp, no_of_samples);
  init_spectrum(asymmetry, no_of_samples);

  for(unsigned int i = 0; i < no_of_samples; ++i)
  {
    double wavelength = i*25.0e-9 + 375.0e-9;
    
    // Adding soluble solids (n=1.351 measured by Benitez et al. [2007], see also Genovese and Lozano [2006])
    // and host absorption as measured in browned clarified apple juice [Beveridge et al. 1986].
    // Blending absorption with water absorption as we get away from the absorption peak 
    // (the curve is too limited in precision when values get close to zero).
    unsigned int abs_idx = i == 0 ? 0 : i - 1;
    if(abs_idx > 12) abs_idx = 12;
    double host_abs = apple_juice_abs[2][abs_idx]/2.0e-2*log(10.0);
    double n_imag = host_abs*wavelength/(4.0*M_PI);
    double x = abs_idx/12.0;
    refrac_host[i] = complex<double>(refrac_water[i].real() + (1.351 - 1.333), n_imag*(1.0 - x) + refrac_water[i].imag()*x);
    
    // Adding apple absorption [Saeys et al. 2008, Lu et al. 2010] to particle refractive index
    double abs_coef;
    if(i < 6)
    {
      double lambda = wavelength*1.0e9;
      unsigned int idx = cdf_bsearch(lambda, apple_abs[0], no_of_samples) - 1;
      double pos = (lambda - apple_abs[0][idx])/(apple_abs[0][idx + 1] - apple_abs[0][idx]);
      abs_coef = (apple_abs[1][idx]*(1.0 - pos) + apple_abs[1][idx + 1]*pos)*1.0e3;
    }
    else
      abs_coef = bruised_apple_abs[0][(i - 5)*5]*1.0e2;
    refrac_apple[i] = complex<double>(apple.refrac_idx.real(), abs_coef*wavelength/(4.0*M_PI));

    //cerr << "Wavelength: " << wavelength << " (idx: " << i << ")" << endl;
    //cerr << "Refractive Index of host medium: " << refrac_host[i] << endl;
    //cerr << "Refractive index of apple particles: " << refrac_apple[i] << endl;

    optical_props_log10(&apple, wavelength, refrac_host[i], &refrac_apple[i]);
    //cerr << "Apple properties: " << apple.ext << " " << apple.sca << " " << apple.abs << " " << apple.g << endl;

    double host_absorp = 4.0*M_PI*refrac_host[i].imag()/wavelength;
    //cerr << "Host extinction: " << host_absorp << endl;

    extinct[i] = host_absorp + apple.ext;
    scatter[i] = apple.sca;
    absorp[i] = extinct[i] - scatter[i];
    ior[i] = complex<double>(refrac_host[i].real(), absorp[i]*wavelength/(4.0*M_PI));
    if(scatter[i] != 0.0)
      asymmetry[i] = (apple.sca*apple.g)/scatter[i];

    //cout << "Extinction coefficient (" << wavelength*1e9 << "nm): " << extinct[i] << endl
    //     << "Scattering coefficient (" << wavelength*1e9 << "nm): " << scatter[i] << endl
    //     << "Absorption coefficient (" << wavelength*1e9 << "nm): " << absorp[i] << endl
    //     << "Ensemble asymmetry parameter (" << wavelength*1e9 << "nm):    " << asymmetry[i] << endl;

    //if(i < 16)
    //{
    //  cout << endl << "<press enter for next set of values>" << endl;
    //  cin.get();
    //}
  }
  m.name = "AppleJuice";
  m.turbid = C > 0.0;
  return m;
}
