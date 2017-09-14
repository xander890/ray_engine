#include "cputimer.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include<windows.h>
#include<mmsystem.h>

// inv_freq is 1 over the number of ticks per second.
static double inv_freq;
static int freq_initialized = 0;
static int use_high_res_timer = 0;

double currentTime()
{
	if (!freq_initialized) {
		LARGE_INTEGER freq;
		use_high_res_timer = QueryPerformanceFrequency(&freq);
		inv_freq = 1.0 / freq.QuadPart;
		freq_initialized = 1;
	}
	if (use_high_res_timer) {
		LARGE_INTEGER c_time;
		if (QueryPerformanceCounter(&c_time)) {
			return c_time.QuadPart*inv_freq;
		}
		else {
			return -1.0;
		}
	}
	else {
		return ((double)timeGetTime()) * 1.0e-3;
	}
	return -1.0;
}

#else

double currentTime()
{
	struct timeval tv;
	if (gettimeofday(&tv, 0)) {
		fprintf(stderr, "sutilCurrentTime(): gettimeofday failed!\n");
		return -1.0;
	}

	return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

#endif