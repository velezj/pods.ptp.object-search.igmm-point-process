
add_executable( test-igmm-mcmc
  test-igmm-mcmc.cpp )
pods_use_pkg_config_packages( test-igmm-mcmc math-core point-process-core probability-core igmm-point-process lcmtypes_math-core lcmtypes_point-process-core lcmtypes_probability-core lcmtypes_igmm-point-process )
pods_install_executables( test-igmm-mcmc )



add_executable( test-igmm-line-gaussian-poisson
  test-igmm-line-gaussian-poisson.cpp )
pods_use_pkg_config_packages( test-igmm-line-gaussian-poisson 
  math-core lcmtypes_math-core
  point-process-core lcmtypes_point-process-core
  probability-core lcmtypes_probability-core
  probability-core-graphics
  igmm-point-process  lcmtypes_igmm-point-process
  point-process-experiment-core
  )
pods_install_executables( test-igmm-line-gaussian-poisson )


add_executable( test-igmm-samples
  test-igmm-samples.cpp )
pods_use_pkg_config_packages( test-igmm-samples
  math-core lcmtypes_math-core
  point-process-core lcmtypes_point-process-core
  probability-core lcmtypes_probability-core
  igmm-point-process  lcmtypes_igmm-point-process
  point-process-experiment-core
  )
pods_install_executables( test-igmm-samples )


add_executable( test-igmm-entropy
  test-entropy.cpp )
pods_use_pkg_config_packages( test-igmm-entropy
  math-core lcmtypes_math-core
  point-process-core lcmtypes_point-process-core
  probability-core lcmtypes_probability-core
  igmm-point-process  lcmtypes_igmm-point-process
  point-process-experiment-core
  )
pods_install_executables( test-igmm-entropy )


add_executable( igmm-point-process-test-negative-observation-likelihood-for-mean
  test-negative-observation-likelihood-for-mean.cpp )
pods_use_pkg_config_packages( igmm-point-process-test-negative-observation-likelihood-for-mean
  math-core lcmtypes_math-core
  point-process-core lcmtypes_point-process-core
  probability-core lcmtypes_probability-core
  probability-core-graphics
  igmm-point-process  lcmtypes_igmm-point-process
  point-process-experiment-core
  )
pods_install_executables( igmm-point-process-test-negative-observation-likelihood-for-mean )


add_subdirectory( unit-tests )