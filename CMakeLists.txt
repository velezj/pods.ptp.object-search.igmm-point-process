cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME object-search.igmm-point-process)
include(cmake/pods.cmake)

add_definitions( -std=c++0x -Wall -fdiagnostics-show-option -Wno-unused-local-typedefs -fPIC )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)



add_library( object-search.igmm-point-process SHARED
  src/model.cpp
  src/mcmc.cpp
  src/igmm_point_process.cpp
)
pods_install_headers(
  src/model.hpp
  src/mcmc.hpp
  src/igmm_point_process.hpp
  DESTINATION
  igmm-point-process)
pods_install_libraries( object-search.igmm-point-process )
pods_use_pkg_config_packages(object-search.igmm-point-process 
    object-search.probability-core 
    gsl-1.16 
    object-search.math-core 
    boost-1.54.0)
pods_install_pkg_config_file(object-search.igmm-point-process
    CFLAGS
    LIBS -lobject-search.igmm-point-process
    REQUIRES object-search.math-core object-search.probability-core object-search.point-process-core boost-1.54.0 gsl-1.16
    VERSION 0.0.1)


#add_subdirectory( test )
