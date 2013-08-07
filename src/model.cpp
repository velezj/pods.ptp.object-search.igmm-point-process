
#include "model.hpp"
#include "mcmc.hpp"
#include <math-core/io.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>


using namespace math_core;
using namespace probability_core;

namespace igmm_point_process {


  //======================================================================

  std::vector<nd_point_t> 
  points_for_mixture( const igmm_point_process_state_t& state,
		      const size_t mixture_i )
  {
    
    std::vector<nd_point_t> points;
    std::vector<size_t> indices = state.mixture_to_observation_indices[ mixture_i ];
    for( size_t i = 0; i < indices.size(); ++i ) {
      points.push_back( state.observations[ indices[i] ] );
    }
    
    return points;
  }


  //======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const igmm_point_process_state_t& state )
  {
    os << "IGMM STATE: " << std::endl;
    os << state.model << std::endl;
    os << "  observations: " << std::endl;
    for( size_t i = 0; i < state.observations.size(); ++i ) {
      os << "    " << state.observations[i] << " -> " << state.observation_to_mixture[i] << std::endl;
    }
    os << std::endl;
    
    for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {
      os << "  Mixture " << mixture_i << ": " << std::endl;
      std::vector<nd_point_t> points = points_for_mixture( state, mixture_i );
      for( size_t i = 0; i < points.size(); ++i ) {
	os << "      point[" << i << "]: " << points[i] << std::endl;
      }
      os << "    gaussian: " << state.mixture_gaussians[ mixture_i ] << std::endl;
      os << "    poisson : " << state.mixture_poissons[ mixture_i ] << std::endl;
    }
    
    return os;
  }

  //======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const igmm_point_process_model_t& model )
  {
    os << "model alpha: " << model.alpha << std::endl;
    os << "model mean: " << model.mean_distribution << std::endl;
    os << "model precision: " << model.precision_distribution << std::endl;
    os << "model num-points: " << model.num_points_per_gaussian_distribution << std::endl;
    return os;
  }

  //======================================================================

  std::vector<nd_point_t> sample_from( const igmm_point_process_state_t& state )
  {
    
    std::vector<nd_point_t> points;
    
    // sample points from each mixture separately
    for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {

      poisson_distribution_t num_distribution = state.mixture_poissons[ mixture_i ];
      gaussian_distribution_t spread_distribution = state.mixture_gaussians[ mixture_i ];
      
      // sample a number of points
      unsigned int num_points = sample_from( num_distribution );
      
      // now for each point, sample it's location from the spread
      for( size_t p_i = 0; p_i < num_points; ++p_i ) {
	nd_point_t p = sample_from( spread_distribution );
	points.push_back( p );
      }

    }

    // sample from a whole new mixture with some probability
    double new_mixture_probability =
      state.model.alpha / 
      ( state.model.alpha + state.mixture_gaussians.size() - 1 );
    if( flip_coin( new_mixture_probability ) ) {
      
      // sample the new mixture from the priors
      gaussian_distribution_t spread_distribution = 
	sample_gaussian_from( state.model.mean_distribution,
			      state.model.precision_distribution );
      poisson_distribution_t num_distribution = 
	sample_poisson_from( state.model.num_points_per_gaussian_distribution );
      
      // sample a number of points
      unsigned int num_points = sample_from( num_distribution );
      
      // now for each point, sample it's location from the spread
      for( size_t p_i = 0; p_i < num_points; ++p_i ) {
	nd_point_t p = sample_from( spread_distribution );
	points.push_back( p );
      }
    }

    // Ok, now add all of the points already known
    points.insert( points.end(), state.observations.begin(),
		   state.observations.end() );
    
    return points;
  }

  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================


}
