#if !defined( __IGMM_POINT_PROCESS_IGMM_POINT_PROCESS_HPP__ )
#define __IGMM_POINT_PROCESS_IGMM_POINT_PROCESS_HPP__


#include <point-process-core/point_process.hpp>
#include "model.hpp"
#include "mcmc.hpp"
#include <probability-core/distributions.hpp>

namespace igmm_point_process {

  // Description:
  // The mcmc_point_process_t subclass for the igmm_point_process
  class igmm_point_process_t : public point_process_core::mcmc_point_process_t
  {

  public: // CREATION

    // Description:
    // Create a new process
    igmm_point_process_t( const math_core::nd_aabox_t& window,
			  const igmm_point_process_model_t& model,
			  const std::vector<math_core::nd_point_t>& obs )
    {
      _window = window;
      _state.model = model;
      _state.observations = obs;
      for( std::size_t i = 0; i < _state.observations.size(); ++i ) {
	probability_core::gaussian_distribution_t gauss 
	  = model.mean_distribution;
	for( std::size_t c_i = 0; c_i < _state.observations[i].n; ++c_i ) {
	  gauss.means[c_i] = _state.observations[i].coordinate[c_i];
	}
	_state.mixture_gaussians.push_back( gauss );
	probability_core::poisson_distribution_t pos;
	pos.lambda = sample_from( _state.model.num_points_per_gaussian_distribution );
	_state.mixture_poissons.push_back( pos );
	_state.observation_to_mixture.push_back( i );
	std::vector<std::size_t> indices = std::vector<std::size_t>( 1, i );
	_state.mixture_to_observation_indices.push_back( indices );
      }
      
    }


    // Description:
    // Clones this process
    virtual
    boost::shared_ptr<mcmc_point_process_t>
    clone() const
    {
      return boost::shared_ptr<mcmc_point_process_t>( new igmm_point_process_t( _window, _state ) );
    }

  public: // API

    // Description:
    // Retrusn the window for this point process
    virtual
    math_core::nd_aabox_t window() const
    {
      return _window;
    }


    // Description:
    virtual
    std::vector<math_core::nd_point_t>
    observations() const
    {
      return _state.observations;
    }

    // Description:
    // Sample ffrom the process
    virtual
    std::vector<math_core::nd_point_t>
    sample() const
    {
      return sample_from( _state );
    }
    
    // Description:
    // Run a single mcmc step
    virtual
    void single_mcmc_step()
    {
      mcmc_single_step( _state );
    }
    
    // Description:
    // Add observations to this process
    virtual
    void add_observations( const std::vector<math_core::nd_point_t>& obs )
    {
      using namespace probability_core;
      _state.observations.insert( _state.observations.end(),
				  obs.begin(),
				  obs.end() );
      for( std::size_t o_i = 0;
	   o_i < obs.size();
	   ++o_i ) {
	gaussian_distribution_t gauss = _state.model.mean_distribution;
	gauss.means = obs[o_i].coordinate;
	_state.mixture_gaussians.push_back( gauss );
	poisson_distribution_t pos;
	pos.lambda = 2;
	_state.mixture_poissons.push_back( pos );
	_state.observation_to_mixture.push_back( _state.mixture_gaussians.size() - 1 );
	std::vector<std::size_t> indices = std::vector<std::size_t>( 1, o_i );
	_state.mixture_to_observation_indices.push_back( indices );
      }
    }


    // Descripiton:
    // Add a negative observation
    virtual
    void add_negative_observation( const math_core::nd_aabox_t& region )
    {
      _state.negative_observations.push_back( region );
    }


    // Description:
    // Turns on or off mcmc  tracing.
    // CURRENTLY DOES NOTHING
    virtual
    void trace_mcmc( const std::string& trace_dir )
    {
    }
    
    virtual 
    void trace_mcmc_off()
    {
    }

    virtual
    void print_shallow_trace( std::ostream& out ) const
    {
      out << "EEEK";
    }

    virtual
    void print_model_shallow_trace( std::ostream& out ) const
    {
      out << "EEEK::MODEL";
    }


  public: // STATE

    // Description:
    // The window for this point process
    math_core::nd_aabox_t _window;

    
    // Description:
    // The igmm-point-process-state for this process
    igmm_point_process_state_t _state;

  protected: // CREATION
    
    // Description:
    // Creates a new point process with copy of the given state
    igmm_point_process_t( const nd_aabox_t& window,
			  const igmm_point_process_state_t& state )
      : _window( window ), _state( state )
    {}

  };
  


}

#endif

