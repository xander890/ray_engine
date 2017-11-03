cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

################### Variables. ####################
# Change if you want modify path or other values. #
###################################################

set(PROJECT_NAME nfd)
# Output Variables
set(OUTPUT_DEBUG lib64/)
set(OUTPUT_REL lib64/)
# Folders files
set(CPP_DIR_1 ./)
set(CPP_DIR_2 ./)
set(HEADER_DIR_1 )
set(HEADER_DIR_2 include)

############## CMake Project ################
#        The main options of project        #
#############################################

project(${PROJECT_NAME} CXX)

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

