#include "brdf_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <logger.h>

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

bool read_brdf(const char *filename, int& n, double*& brdf)
{
    FILE *f;
    fopen_s(&f, filename, "rb");
	if (!f)
		return false;

	int dims[3];
	fread(dims, sizeof(int), 3, f);
	n = 3 * dims[0] * dims[1] * dims[2];

	brdf = new double[n];
	fread(brdf, sizeof(double), n, f);


	fclose(f);
	return true;
}

// Moves the content from a double to a float array. 
// Loses precision and the old array.
template<typename T, typename S>
void convert_array_type(int size, std::vector<T>& new_array, S*& old_array)
{
	//Note that we need to loop in order to tightly pack the new array.
	new_array.resize(size);
	for (int i = 0; i < size; i++)
	{
		new_array[i] = static_cast<T>(old_array[i]);
	}
	delete[] old_array;
}

void read_brdf_f(const std::string& merl_folder, const std::string& name, std::vector<float> & brdf_f)
{
    double * brdf_d;
    int size;
	std::string file = merl_folder + name;
    if (read_brdf(file.c_str(), size, brdf_d))
    {
        convert_array_type(size, brdf_f, brdf_d);
        auto c = file.find_last_of(".");
        file.erase(c);
        Logger::info << "Max: " << std::to_string(*std::max_element(brdf_f.begin(), brdf_f.end())) << std::endl;
    }
}


