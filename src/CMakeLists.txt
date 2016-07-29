cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_LIST_DIR})
  if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting CMAKE_BUILD_TYPE type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
  endif()
  project(fftset VERSION 0.1.0 LANGUAGES C)
endif()

add_library(fftset STATIC fftset.c fftset_mod_freqoffsetreal.c fftset_vec.c fftset_mod_cplx.c ../fftset.h)
target_include_directories(fftset PRIVATE "../..")
target_link_libraries(fftset cop)
