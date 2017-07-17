#ifndef MT_RANDOM_H
#define MT_RANDOM_H

#include "Randomizer.h"
extern Randomizer randomizer;

// generates a random number on [0,1]-real-interval
inline double mt_random()
{
  return randomizer.mt_random();
}

// generates a random number on [0,1)-real-interval
inline double mt_random_half_open()
{
  return randomizer.mt_random_half_open();
}

// generates a random number on (0,1)-real-interval
inline double mt_random_open()
{
  return randomizer.mt_random_open();
}

// Use the following function in constructors that 
// could be used for global variable instances.
inline double safe_mt_random()
{
  return randomizer.safe_mt_random();
}

#endif // MT_RANDOM_H
