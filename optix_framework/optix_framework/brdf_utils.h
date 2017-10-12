#ifndef BRDF_UTILS_H
#define BRDF_UTILS_H

#include "folders.h"
#include <vector>
#include <map>
// Read BRDF data
bool read_brdf(const char *filename, int& size, double*& brdf);
void read_all_brdfs(std::map<std::string, std::vector<float>*>& brdfs);
void read_all_brdfs(const char * merl_folder, const char * merl_database, std::map<std::string, std::vector<float>*>& brdfs);
void get_merl_brdf_list(const char* merl_database, std::vector<std::string>& brdfs);
void read_brdf_f(const std::string& merl_folder, const std::string& name, std::vector<float> & brdf_f);
 
#endif // !BRDF_UTILS_H
