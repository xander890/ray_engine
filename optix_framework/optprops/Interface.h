// 02576 Rendering Framework
// Written by Jeppe Revall Frisvad, 2009
// Copyright (c) DTU Informatics 2009

#ifndef INTERFACE_H
#define INTERFACE_H

#include <string>
#include "Medium.h"

class Interface
{
public:
  Interface() : med_in(0), med_out(0) { }

  OPTPROPS_API std::string get_name();
  OPTPROPS_API void set_name(const std::string & name);
  Medium* med_in;
  Medium* med_out;
private:
	std::string name;
};

#endif
