#pragma once

#ifdef OPTPROPS_EXPORT  
#define OPTPROPS_API __declspec(dllexport)   
#else  
#define OPTPROPS_API __declspec(dllimport)   
#endif  