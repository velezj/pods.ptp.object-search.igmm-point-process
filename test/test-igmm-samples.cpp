
#include <point-process-experiment-core/simulated_data.hpp>
#include <math-core/io.hpp>
#include <math-core/matrix.hpp>
#include <igmm-point-process/mcmc.hpp>
#include <iostream>

using namespace math_core;
using namespace probability_core;
using namespace point_process_experiment_core;
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
      3.0 );

  
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
  
  // default state with each observation it's own mixture
  igmm_point_process_state_t state;
  state.model = model;
  state.observations = points;
  for( size_t i = 0; i < state.observations.size(); ++i ) {
    gaussian_distribution_t gauss = model.mean_distribution;
    gauss.means[0] = state.observations[i].coordinate[0];
    state.mixture_gaussians.push_back( gauss );
    poisson_distribution_t pos;
    pos.lambda = 2;
    state.mixture_poissons.push_back( pos );
    state.observation_to_mixture.push_back( i );
    std::vector<size_t> indices = std::vector<size_t>( 1, i );
    state.mixture_to_observation_indices.push_back( indices );
  }


  // draw samples from it
  for( int i = 0; i < 10; ++i ) {
    std::vector<nd_point_t> sample = sample_from( state );
    std::cout << "sample: #point = " << sample.size() << std::endl;
  }

}
