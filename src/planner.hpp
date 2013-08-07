
#if !defined( __IGMM_POINT_PROCESS_PLANNER_HPP__ )
#defined __IGMM_POINT_PROCESS_PLANNER_HPP__

#include "model.hpp"
#include <point-process-core/marked_grid.hpp>
#include <point-process-core/entropy.hpp>

namespace igmm_point_process {

  using namespace point_process_core;
  
  // Description:
  // Parameters used for planner
  struct planner_parameters_t 
  {
    entropy_estimator_parameters_t entropy_params;
    unsigned long burnin_mcmc_iterations;
    unsigned long update_model_mcmc_iterations;
    unsigned long num_observation_samples_to_estimate_post_observation_entropy;
  };


  // Description:
  // Given an observation grid (with marks on unavailable locations)
  // Returns the next best grid cell to take an observations at
  marked_grid_cell_t
  choose_next_observation_using_greedy_entropy_reduction
  ( igmm_point_process_state_t& state,
    const planner_parameters_t& params,
    const marked_grid_t& observation_grid );


  // Description:
  // Given a state and an observation cell,
  // returns the expected entropy of the state after taking the
  // observations.
  double
  expected_entropy_after_observations
  ( const igmm_point_process_state_t& state,
    const marked_grid_cell_t& cellm
    const marked_grid_t& observation_grid,
    const planner_parameters_t& params );


}

#endif
