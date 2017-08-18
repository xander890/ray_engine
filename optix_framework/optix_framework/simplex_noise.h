#include <vector>

// THIS CLASS IS A TRANSLATION TO C++11 FROM THE REFERENCE
// JAVA IMPLEMENTATION OF THE IMPROVED PERLIN FUNCTION (see http://mrl.nyu.edu/~perlin/noise/)
// THE ORIGINAL JAVA IMPLEMENTATION IS COPYRIGHT 2002 KEN PERLIN

// I ADDED AN EXTRA METHOD THAT GENERATES A NEW PERMUTATION VECTOR (THIS IS NOT PRESENT IN THE ORIGINAL IMPLEMENTATION)

#ifndef SIMPLEXNOISE_H
#define SIMPLEXNOISE_H

class SimplexNoise {
    // The permutation vector
    std::vector<int> perm;
public:
    // Initialize with the reference values for the permutation vector
    SimplexNoise();
    // Generate a new permutation vector based on the value of seed
    // Get a noise value, for 2D images z can have any value
    double noise(double x, double y, double z, double w) const;
};

#endif
