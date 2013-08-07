
#include <point-process-experiment-core/simulated_data.hpp>
#include <math-core/io.hpp>
#include <math-core/matrix.hpp>
#include <igmm-point-process/igmm_point_process.hpp>
#include <point-process-core/entropy.hpp>
#include <iostream>

using namespace math_core;
using namespace probability_core;
using namespace point_process_experiment_core;
using namespace point_process_core;
using namespace igmm_point_process;


int main( int argc, char** argv )
{

  // craete the window (the range)
  nd_aabox_t window;
  window.n = 1;
  window.start = point( 0 );
  window.end = point( 100 );

  // simulate some points
  std::vector<nd_point_t> points 
    = simulate_line_point_clusters_gaussian_spread_poisson_size
    ( window,
      4,
      0.1,
      2.0 );

  
  // now create a new point process;
  igmm_point_process_model_t model;
  model.alpha = 1;
  model.mean_distribution.dimension = 1;
  model.mean_distribution.means.push_back( 10 );
  model.mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * 1.0 );
  model.precision_distribution.shape = 2;
  model.precision_distribution.rate = 0.5;
  model.num_points_per_gaussian_distribution.shape = 2;
  model.num_points_per_gaussian_distribution.rate = 0.5;

  boost::shared_ptr<mcmc_point_process_t> 
    proc( new igmm_point_process_t(window,
				   model,
				   points ) );
  

  // parameterts for entropy compuitation (default params )
  entropy_estimator_parameters_t entropy_params;
  entropy_params.num_samples = 1000;
  entropy_params.num_samples_to_skip = 1;
  entropy_params.histogram_grid_cell_size = 10.0;

  // run some mcmc and compute entropy over time
  for( size_t i = 0; i < 100; ++i ) {
    double entropy =
      estimate_entropy_from_samples( entropy_params,
				     proc );
    std::cout << "entropy:      " << entropy << std::endl;
  }
}
