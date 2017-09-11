#include "logger.h"


const char * Logger::BLUE = "\x1b[36m";
const char * Logger::YELLOW = "\x1b[33m";
const char * Logger::GREEN = "\x1b[32m";
const char * Logger::RED = "\x1b[31m";
const char * Logger::RESET = "\x1b[0m";
bool Logger::is_color_enabled = false;

Logger Logger::info = Logger(std::cout, BLUE, "info");
Logger Logger::error = Logger(std::cout, RED, "error");
Logger Logger::warning = Logger(std::cout, YELLOW, "warning");
Logger Logger::debug = Logger(std::cout, GREEN, "debug");