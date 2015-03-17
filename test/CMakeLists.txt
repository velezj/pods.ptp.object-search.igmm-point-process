
add_executable( object-search.test-igmm-mcmc
  test-igmm-mcmc.cpp )
pods_use_pkg_config_packages( object-search.test-igmm-mcmc 
    object-search.math-core 
    object-search.point-process-core 
    object-search.probability-core 
    object-search.igmm-point-process 
    lcmtypes_object-search.math-core 
     
    lcmtypes_object-search.probability-core 
    lcmtypes_object-search.igmm-point-process )
pods_install_executables( object-search.test-igmm-mcmc )



add_executable( object-search.test-igmm-line-gaussian-poisson
  test-igmm-line-gaussian-poisson.cpp )
pods_use_pkg_config_packages( object-search.test-igmm-line-gaussian-poisson 
  object-search.math-core lcmtypes_object-search.math-core
  object-search.point-process-core 
  object-search.probability-core lcmtypes_object-search.probability-core
  object-search.probability-core-graphics
  object-search.igmm-point-process  lcmtypes_object-search.igmm-point-process
  object-search.point-process-experiment-core
  )
pods_install_executables( object-search.test-igmm-line-gaussian-poisson )


add_executable( object-search.test-igmm-samples
  test-igmm-samples.cpp )
pods_use_pkg_config_packages( object-search.test-igmm-samples
  object-search.math-core lcmtypes_object-search.math-core
  object-search.point-process-core 
  object-search.probability-core lcmtypes_object-search.probability-core
  object-search.igmm-point-process  lcmtypes_object-search.igmm-point-process
  object-search.point-process-experiment-core
  )
pods_install_executables( object-search.test-igmm-samples )


add_executable( object-search.test-igmm-entropy
  test-entropy.cpp )
pods_use_pkg_config_packages( object-search.test-igmm-entropy
  object-search.math-core lcmtypes_object-search.math-core
  object-search.point-process-core 
  object-search.probability-core lcmtypes_object-search.probability-core
  object-search.igmm-point-process  lcmtypes_object-search.igmm-point-process
  object-search.point-process-experiment-core
  )
pods_install_executables( object-search.test-igmm-entropy )


add_executable( object-search.igmm-point-process-test-negative-observation-likelihood-for-mean
  test-negative-observation-likelihood-for-mean.cpp )
pods_use_pkg_config_packages( object-search.igmm-point-process-test-negative-observation-likelihood-for-mean
  object-search.math-core lcmtypes_object-search.math-core
  object-search.point-process-core 
  object-search.probability-core lcmtypes_object-search.probability-core
  object-search.probability-core-graphics
  object-search.igmm-point-process  lcmtypes_object-search.igmm-point-process
  object-search.point-process-experiment-core
  )
pods_install_executables( object-search.igmm-point-process-test-negative-observation-likelihood-for-mean )


add_subdirectory( unit-tests )
