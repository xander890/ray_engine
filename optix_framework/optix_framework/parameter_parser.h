#ifndef parameter_parser_h__
#define parameter_parser_h__

#include "parserstringhelpers.h"

#include "logger.h"
#include <map>


class ParameterParser
{
	struct NodeElem
	{
		std::string value;
		std::string comment;
	};

public:	
	static void init(const std::string & document_f);
	static void free();

	static void dump_used_parameters(const std::string & document_f);
	static void override_parameters(std::vector<std::string> & parameters);

	template <typename T>
	static T get_parameter(const char * tag, const char * n, T default_value, const char * comment = "")
	{
		T val;
		if (parameters.count(tag) != 0 && parameters[tag].count(n) != 0)
		{
			val = tovalue<T>(parameters[tag][n].value);
			used_parameters[tag][n] = parameters[tag][n];
			used_parameters[tag][n].comment = std::string(comment);
		}
		else
		{
			val = default_value;
			used_parameters[tag][n] = { tostring<T>(val) , std::string(comment) };
		}
		return val;
	}

	template <typename T>
	static T get_parameter(const std::string & tag, const std::string &  n, T default_value, const std::string & comment = "")
	{
		return get_parameter(tag.c_str(), n.c_str(), default_value, comment.c_str());
	}
protected:
	ParameterParser(void);
	~ParameterParser(void);

	static std::string document_file;

	static std::map<std::string, std::map<std::string, NodeElem>> parameters;
	static std::map<std::string, std::map<std::string, NodeElem>> used_parameters;
	
	static void parse_doc();
};
#endif // parameter_parser_h__
