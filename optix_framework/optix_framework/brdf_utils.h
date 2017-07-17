#ifndef BRDF_UTILS_H
#define BRDF_UTILS_H

#include "folders.h"
#include <vector>
#include <map>
// Read BRDF data
bool read_brdf(const char *filename, int& size, double*& brdf);
void read_all_brdfs(std::map<std::string, std::vector<float>*>& brdfs);
void read_all_brdfs(const char * merl_folder, const char * merl_database, std::map<std::string, std::vector<float>*>& brdfs);
#endif // !BRDF_UTILS_H
