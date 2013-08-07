
add_executable( igmm-point-process-unit-test-main
  unit-test-main.cpp)
pods_use_pkg_config_packages( igmm-point-process-unit-test-main 
  unittest++
  math-core lcmtypes_math-core
  point-process-core lcmtypes_point-process-core
  probability-core lcmtypes_probability-core
  igmm-point-process  lcmtypes_igmm-point-process
  point-process-experiment-core )
pods_install_executables( igmm-point-process-unit-test-main )

