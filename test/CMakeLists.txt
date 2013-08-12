
add_executable( p2l-test-igmm-mcmc
  test-igmm-mcmc.cpp )
pods_use_pkg_config_packages( p2l-test-igmm-mcmc 
    p2l-math-core 
    p2l-point-process-core 
    p2l-probability-core 
    p2l-igmm-point-process 
    lcmtypes_p2l-math-core 
     
    lcmtypes_p2l-probability-core 
    lcmtypes_p2l-igmm-point-process )
pods_install_executables( p2l-test-igmm-mcmc )



add_executable( p2l-test-igmm-line-gaussian-poisson
  test-igmm-line-gaussian-poisson.cpp )
pods_use_pkg_config_packages( p2l-test-igmm-line-gaussian-poisson 
  p2l-math-core lcmtypes_p2l-math-core
  p2l-point-process-core 
  p2l-probability-core lcmtypes_p2l-probability-core
  p2l-probability-core-graphics
  p2l-igmm-point-process  lcmtypes_p2l-igmm-point-process
  p2l-point-process-experiment-core
  )
pods_install_executables( p2l-test-igmm-line-gaussian-poisson )


add_executable( p2l-test-igmm-samples
  test-igmm-samples.cpp )
pods_use_pkg_config_packages( p2l-test-igmm-samples
  p2l-math-core lcmtypes_p2l-math-core
  p2l-point-process-core 
  p2l-probability-core lcmtypes_p2l-probability-core
  p2l-igmm-point-process  lcmtypes_p2l-igmm-point-process
  p2l-point-process-experiment-core
  )
pods_install_executables( p2l-test-igmm-samples )


add_executable( p2l-test-igmm-entropy
  test-entropy.cpp )
pods_use_pkg_config_packages( p2l-test-igmm-entropy
  p2l-math-core lcmtypes_p2l-math-core
  p2l-point-process-core 
  p2l-probability-core lcmtypes_p2l-probability-core
  p2l-igmm-point-process  lcmtypes_p2l-igmm-point-process
  p2l-point-process-experiment-core
  )
pods_install_executables( p2l-test-igmm-entropy )


add_executable( p2l-igmm-point-process-test-negative-observation-likelihood-for-mean
  test-negative-observation-likelihood-for-mean.cpp )
pods_use_pkg_config_packages( p2l-igmm-point-process-test-negative-observation-likelihood-for-mean
  p2l-math-core lcmtypes_p2l-math-core
  p2l-point-process-core 
  p2l-probability-core lcmtypes_p2l-probability-core
  p2l-probability-core-graphics
  p2l-igmm-point-process  lcmtypes_p2l-igmm-point-process
  p2l-point-process-experiment-core
  )
pods_install_executables( p2l-igmm-point-process-test-negative-observation-likelihood-for-mean )


add_subdirectory( unit-tests )
