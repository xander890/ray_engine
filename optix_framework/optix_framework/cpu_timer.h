#pragma once
#include <chrono>
#include <ratio>

// Simple system call to get the most accurate timing possible.
std::chrono::time_point<std::chrono::high_resolution_clock> currentTime();