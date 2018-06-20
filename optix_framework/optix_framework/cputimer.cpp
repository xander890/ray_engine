#include "cputimer.h"

std::chrono::time_point<std::chrono::high_resolution_clock> currentTime()
{
	return std::chrono::high_resolution_clock::now();
}
