#pragma once
#ifndef LOGGER_H
#define LOGGER_H
#include <iostream>
#include <string>

class Logger {

#define USE_COLORS
#ifdef USE_COLORS
#define COLOR_SELECTOR(x) x
#else
#define COLOR_SELECTOR(x)
#endif

public:
	explicit Logger(std::ostream& _out, const char * _color_symbol, const char * _color_string) : out(_out), color_symbol(_color_symbol), color_string(_color_string) {}

	template<typename T>
	std::string get_str(const T&  v) { return std::to_string(v); }

	template<>
	std::string get_str(const std::string & v) { return v; }

	template<typename T>
	Logger& operator<<(const T v)  
	{ 
		if (start_of_line)
		{
			out << COLOR_SELECTOR(color_symbol << ) "[" << color_string << "] " COLOR_SELECTOR(<< RESET);
			start_of_line = false;
		}
		std::string a = get_str<T>(v);
		out << a;
		if (a.back() == '\n')
			start_of_line = true;
		return *this; 
	}

	template<> Logger& operator<<<const char*>(const char* v)
	{
		return this->operator<<<std::string>(std::string(v));
	}


	Logger& operator<<(std::ostream& (*F)(std::ostream&)) { F(out); start_of_line = true;  return *this; }

	static Logger info;
	static Logger debug;
	static Logger error;
	static Logger warning;

protected:
	std::ostream& out;
	std::string color_symbol;
	std::string color_string;


	bool start_of_line = true;

private:
	static const char * BLUE; 
	static const char * RED;
	static const char * GREEN;
	static const char * RESET; 
	static const char * YELLOW;
};



#endif