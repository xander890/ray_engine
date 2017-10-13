#pragma once
#include <string>

// Prettifies an enum string. TEST_MY_STRING becomes Test my string.
inline std::string prettify(const std::string & s)
{
	std::string res = s;
	for (unsigned int i = 0; i <= s.length(); i++)
	{
		if (s[i] == '_')
		{
			res[i] = ' ';
			continue;
		}
		res[i] = i > 0 ? tolower(s[i]) : toupper(s[i]);
	}
	return res;
}