
add_executable( p2l-igmm-point-process-unit-test-main
  unit-test-main.cpp)
pods_use_pkg_config_packages( p2l-igmm-point-process-unit-test-main 
  unittest++
  p2l-math-core lcmtypes_p2l-math-core
  p2l-point-process-core
  p2l-probability-core lcmtypes_p2l-probability-core
  p2l-igmm-point-process  lcmtypes_p2l-igmm-point-process
  p2l-point-process-experiment-core )
pods_install_executables( p2l-igmm-point-process-unit-test-main )

