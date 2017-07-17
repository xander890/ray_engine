#include "brdf_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>


using namespace std;

inline bool read_brdf(const char *filename, int& n, double*& brdf)
{
	FILE *f = fopen(filename, "rb");
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

void read_all_brdfs(const char * merl_folder, const char * merl_database, std::map<std::string, vector<float>*>& brdfs)
{
	ifstream ifs(merl_database);
	if (!ifs.is_open())
	{
		cout << "Could not find MERL file database." << endl; 
	}
	std::string line;

	while (std::getline(ifs, line))
	{
		string file = (std::string(merl_folder) + line);

		double * brdf;
		int size;
		if(read_brdf(file.c_str(),size, brdf))
		{
			vector<float> * brdf_f = new vector<float>();
			convert_array_type(size, *brdf_f, brdf);
			auto c = line.find_last_of(".");
			line.erase(c);
			brdfs[line] = brdf_f;
			Logger::info << "Max: "  << to_string(*std::max_element(brdf_f->begin(), brdf_f->end())) << endl;
		}
	}
	ifs.close();
}


void read_all_brdfs(std::map<std::string, vector<float>*>& brdfs)
{
	read_all_brdfs(Folders::merl_folder.c_str(), Folders::merl_database_file.c_str(), brdfs);
}