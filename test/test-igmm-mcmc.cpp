
#include <igmm-point-process/mcmc.hpp>
#include <math-core/io.hpp>
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>


using namespace igmm_point_process;
using namespace math_core;
using namespace probability_core;



void print_state( const igmm_point_process_state_t& state )
{
  // print the state:
  std::cout << "IGMM-STATE: " << std::endl;
  
  std::cout << "  model alpha: " << state.model.alpha << std::endl;
  std::cout << "  model mean: " << state.model.mean_distribution << std::endl;
  std::cout << "  model precision: " << state.model.precision_distribution << std::endl;
  std::cout << "  model num-points: " << state.model.num_points_per_gaussian_distribution << std::endl;
  
  std::cout << "  observations: " << std::endl;
  for( size_t i = 0; i < state.observations.size(); ++i ) {
    std::cout << "    " << state.observations[i] << " -> " << state.observation_to_mixture[i] << std::endl;
  }
  std::cout << std::endl;
  
  for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {
    std::cout << "  Mixture " << mixture_i << ": " << std::endl;
    std::vector<nd_point_t> points = points_for_mixture( state, mixture_i );
    for( size_t i = 0; i < points.size(); ++i ) {
      std::cout << "      point[" << i << "]: " << points[i] << std::endl;
    }
    std::cout << "    gaussian: " << state.mixture_gaussians[ mixture_i ] << std::endl;
    std::cout << "    poisson : " << state.mixture_poissons[ mixture_i ] << std::endl;
  }

}

int main( int argc, char** argv )
{


  // create a set of poitns (1d)
  size_t dim = 1;
  std::vector<nd_point_t> points;
  points.push_back( point( 1 ) );
  points.push_back( point( 10 ) );
  points.push_back( point( 20 ) );
  points.push_back( point( 19 ) );
  points.push_back( point( 23 ) );

  // create a state and model
  igmm_point_process_model_t model;
  model.alpha = 1;
  model.mean_distribution.dimension = dim;
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

  
  // do some plaing
  for( size_t i = 0; i < 1000; ++i ) {
    
    mcmc_single_step( state );

    if( i % 100 == 0 ) {
      std::cout << "[" << i << "]--------------------------------------------" << std::endl;
      print_state( state );
      std::cout << "--------------------------------------------" << std::endl;
    }
  }


  print_state( state );

  return 0;
}
