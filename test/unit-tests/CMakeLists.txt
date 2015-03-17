
add_executable( object-search.igmm-point-process-unit-test-main
  unit-test-main.cpp)
pods_use_pkg_config_packages( object-search.igmm-point-process-unit-test-main 
  unittest++
  object-search.math-core lcmtypes_object-search.math-core
  object-search.point-process-core
  object-search.probability-core lcmtypes_object-search.probability-core
  object-search.igmm-point-process  lcmtypes_object-search.igmm-point-process
  object-search.point-process-experiment-core )
pods_install_executables( object-search.igmm-point-process-unit-test-main )

