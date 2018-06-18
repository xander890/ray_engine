/* ----------------------------------------------------------------------- *
 * This file is part of GEL, http://www.imm.dtu.dk/GEL
 * Copyright (C) the authors and DTU Informatics
 * For license and list of authors, see ../../doc/intro.pdf
 * ----------------------------------------------------------------------- */

/**
 * @file Parse.h
 * @brief Parsing various entites from a string.
 */

#ifndef __UTIL_PARSE_H__
#define __UTIL_PARSE_H__

#include <string>
#include <sstream>
#include <vector>
#include "minimal_algebra.h"

namespace Util {
	std::string floatToString(float value);
	void parse(const char* str,bool& x);
	void parse(const char* str,std::string& x);
	void parse(const char* str,int& x);
	void parse(const char* str, Vec2i&);
	void parse(const char* str,float& x);
	void parse(const char* str, Vec2f&);
	void parse(const char* str, Vec3f& vec);
	void parse(const char* str, Vec4f&);
	void parse(const char* str,std::vector<float>& v);
	void parse(const char* str,std::vector<double>& v);
	void parse(const char* str,std::vector<Vec2f>& v);
	void parse(const char* str,std::vector<Vec3f>& v);
	void parse(const char* str,std::vector<int>& v);
	void parseSMPT(const char* str, float& x);
}

#endif // __PARSE_H__
