#include "Interface.h"

std::string Interface::get_name() { return name; }

void Interface::set_name(const std::string & name)
{
	this->name = name;
}
