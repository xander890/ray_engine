#pragma once

#include <vector>
#include <map>

// Read BRDF data
bool read_brdf(const char *filename, int& size, double*& brdf);
void read_brdf_f(const std::string& merl_folder, const std::string& name, std::vector<float> & brdf_f);
 
