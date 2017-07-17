#include "path_tracing.h"
#include "parameter_parser.h"

void PathTracing::init()
{
	int max_splits = ParameterParser::get_parameter<int>("config", "max_splits_rr", 15, "Maximum splits in russian roulette.");
	context["max_splits"]->setInt(max_splits);
	bool use_splitting = ParameterParser::get_parameter<bool>("config", "use_splitting", false, "Use splitting in path tracing.");
	context["use_split"]->setInt(use_splitting);

	unsigned int N = (unsigned int)ParameterParser::get_parameter<int>("config", "N", 1, "Monte carlo samples.");
	context["N"]->setUint(N);
}

void PathTracing::pre_trace()
{
}
