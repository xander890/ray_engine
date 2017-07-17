// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

// Permission to use, copy and modify this software and its documentation without
// fee for educational, research and non-profit purposes, is hereby granted, provided
// that the above copyright notice and the following three paragraphs appear in all copies.

// To request permission to incorporate this software into commercial products contact:
// Vice President of Marketing and Business Development;
// Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or 
// <license@merl.com>.

// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
// HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
// UPDATES, ENHANCEMENTS OR MODIFICATIONS.


#include "BRDF.h"

#include <fstream>
// Read BRDF data
bool read_brdf(const char *filename, double* &brdf)
{
	FILE *f = fopen(filename, "rb");
	if (!f)
		return false;

	int dims[3];
	fread(dims, sizeof(int), 3, f);
	int n = dims[0] * dims[1] * dims[2];
	if (n != BRDF_SAMPLING_RES_THETA_H *
		BRDF_SAMPLING_RES_THETA_D *
		BRDF_SAMPLING_RES_PHI_D / 2)
	{
		fprintf(stderr, "Dimensions don't match\n");
		fclose(f);
		return false;
	}

	brdf = (double*)malloc(sizeof(double) * 3 * n);
	fread(brdf, sizeof(double), 3 * n, f);

	fclose(f);
	return true;
}



int main(int argc, char *argv[])
{
	const char *filename = argv[1];
	double* brdf;

	// read brdf
	if (!read_brdf(filename, brdf))
	{
		fprintf(stderr, "Error reading %s\n", filename);
		exit(1);
	}

	// print out a 16x64x16x64 table table of BRDF values
	const int n = 16;
	for (int i = 0; i < n; i++)
	{
		double theta_in = i * 0.5 * M_PI / n;
		for (int j = 0; j < 4 * n; j++)
		{
			double phi_in = j * 2.0 * M_PI / (4 * n);
			for (int k = 0; k < n; k++)
			{
				double theta_out = k * 0.5 * M_PI / n;
				for (int l = 0; l < 4 * n; l++)
				{
					double phi_out = l * 2.0 * M_PI / (4 * n);
					double red, green, blue;
					lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out, red, green, blue);
					printf("%f %f %f\n", (float)red, (float)green, (float)blue);
				}
			}
		}
	}
	return 0;
}


