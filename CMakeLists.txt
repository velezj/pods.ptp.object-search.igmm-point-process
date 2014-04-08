cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME p2l-igmm-point-process)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()


#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O2 -g")
#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O0 -g3")


add_definitions( -std=c++0x -Wall -fdiagnostics-show-option -Wno-unused-local-typedefs -fPIC )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)



add_library( p2l-igmm-point-process SHARED
  src/model.cpp
  src/mcmc.cpp
)
pods_install_headers(
  src/model.hpp
  src/mcmc.hpp
  src/igmm_point_process.hpp
  DESTINATION
  igmm-point-process)
pods_install_libraries( p2l-igmm-point-process )
pods_use_pkg_config_packages(p2l-igmm-point-process 
    p2l-probability-core 
    gsl-1.16 
    p2l-math-core 
    lcmtypes_p2l-probability-core 
    lcmtypes_p2l-math-core 
    boost-1.54.0)
pods_install_pkg_config_file(p2l-igmm-point-process
    CFLAGS
    LIBS -lp2l-igmm-point-process
    REQUIRES p2l-math-core p2l-probability-core p2l-point-process-core boost-1.54.0 gsl-1.16
    VERSION 0.0.1)


#add_subdirectory( test )
