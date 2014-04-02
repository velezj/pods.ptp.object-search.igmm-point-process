
#if !defined( __IGMM_POINT_PROCESS_MODEL_HPP__ )
#define __IGMM_POINT_PROCESS_MODEL_HPP__

#include <lcmtypes/p2l_math_core.hpp>
#include <lcmtypes/p2l_probability_core.hpp>
#include <probability-core/gaussian.hpp>
#include <iosfwd>

namespace igmm_point_process {

  using namespace math_core;
  using namespace probability_core;

  
  // Description:
  // The parameters for the infinate guassian mixture model point process.
  struct igmm_point_process_model_t{

    double alpha;
    
    gaussian_distribution_t mean_distribution;
    gamma_distribution_t precision_distribution;
    gamma_distribution_t num_points_per_gaussian_distribution;

    double prior_mean;
    double prior_variance;
    
    igmm_point_process_model_t()
      : prior_mean(0),
	prior_variance( 100.0 )
    {}

  };


  // Description:
  // The igmm popint process state
  typedef struct {
    
    // the model parameters
    igmm_point_process_model_t model;
    
    // the observations
    std::vector<nd_point_t> observations;
    
    // the set of gaussian in the mixture
    std::vector< gaussian_distribution_t > mixture_gaussians;
    std::vector< poisson_distribution_t > mixture_poissons;

    // the correspondence between observation and mixture lement
    std::vector< size_t > observation_to_mixture;

    // the mapping between a mixture element and it's observations (as indices)
    std::vector<std::vector< size_t > > mixture_to_observation_indices;

    // the negative observations
    std::vector<nd_aabox_t> negative_observations;

  } igmm_point_process_state_t;


  // Description:
  // Returns the observation points for a partiuclar mixture in a state
  std::vector<nd_point_t> points_for_mixture( const igmm_point_process_state_t& state,
					      const size_t mixture_i );


  // Description:
  // Sample a point cloud from the igmm-point-process.
  std::vector<nd_point_t> sample_from( const igmm_point_process_state_t& state );


  // Description:
  // outputs a state (or model)
  std::ostream& operator<< (std::ostream& os,
			    const igmm_point_process_state_t& state );
  std::ostream& operator<< (std::ostream& os,
			    const igmm_point_process_model_t& model );






}

#endif

