#pragma once
#include "optprops/Medium.h"

// C is the apple particle concentration.
// Values reported in the literature: 0.3 - 2.14 g/L [Beveridge 2002, Genovese and Lozano 2006, Benitez et al. 2007]

inline Medium apple_juice(double C = 1.0);
inline Medium filtered_apple_juice() { return apple_juice(0.0); }