
#include "mcmc.hpp"
#include <probability-core/distribution_utils.hpp>
#include <point-process-core/point_math.hpp>
#include <math-core/matrix.hpp>
#include <limits>
#include <math-core/io.hpp>
#include <math-core/utils.hpp>
#include <iostream>
#include <gsl/gsl_sf_gamma.h>
#include <probability-core/rejection_sampler.hpp>
#include <gsl/gsl_sf_erf.h>
#include <math-core/math_function.hpp>
#include <stdexcept>



namespace igmm_point_process {

  using namespace point_process_core;
  using namespace math_core;
  using namespace probability_core;



  //========================================================================

  // nd_point_t
  // resample_mixture_gaussian_mean( const std::vector<nd_point_t>& points, 
  // 				  const dense_matrix_t& covariance,
  // 				  const gaussian_distribution_t& prior )
  // {

  //   // the dimension of the poitns
  //   std::size_t dim = 0;
  //   if( points.empty() == false )
  //     dim = points[0].n;

  //   // invert the covariance to get a precision
  //   // (Not sure if this actually works for anything greater than 1D poitns!!)
  //   Eigen::MatrixXd prec_mat = to_eigen_mat( covariance ).inverse();

  //   // calculate the sum of the data points
  //   Eigen::VectorXd sum_vec( dim );
  //   for( size_t i = 0; i < points.size(); ++i ) {
  //     for( size_t k = 0; k < dim; ++k ) {
  // 	sum_vec(k) += points[i].coordinate[k];
  //     }
  //   }
    
    
  //   // Invert the prior's covariance to get a prior precision
  //   // and get the prior mean as Eign vector
  //   Eigen::MatrixXd prior_prec = to_eigen_mat( prior.covariance ).inverse();
  //   Eigen::VectorXd prior_mean = to_eigen_mat( prior.means );

  //   // Ok, now calculate the distribution over the new mixture mean
  //   Eigen::MatrixXd new_dist_prec = prec_mat * points.size() + prior_prec;
  //   Eigen::MatrixXd new_dist_cov = new_dist_prec.inverse();
  //   Eigen::VectorXd new_dist_mean 
  //     = new_dist_cov * ( prec_mat * sum_vec + prior_prec * prior_mean );
    
  //   // sample from this gaussian and return the sample
  //   gaussian_distribution_t new_dist;
  //   new_dist.dimension = dim;
  //   new_dist.means = to_vector( new_dist_mean ).component;
  //   new_dist.covariance = to_dense_mat( new_dist_cov );
    
  //   return sample_from( new_dist );
  // }

  //========================================================================

  //========================================================================

  nd_point_t
  resample_mixture_gaussian_mean( const std::vector<nd_point_t>& points, 
				  const std::vector<nd_aabox_t>& negative_observations,
  				  const dense_matrix_t& covariance,
				  const poisson_distribution_t& num_distribution,
  				  const gaussian_distribution_t& prior )
  {

    // create a new posterior function
    boost::shared_ptr<gaussian_mixture_mean_posterior_t>
      posterior( new gaussian_mixture_mean_posterior_t 
		 ( points,
		   negative_observations,
		   covariance,
		   num_distribution,
		   prior ) );


    // set window to posterior with only points mean
    nd_aabox_t window;
    window.n = prior.dimension;
    window.start = point(posterior->posterior_for_points_only.means);
    window.end = point(posterior->posterior_for_points_only.means);

    // now extend the window by at least 3 * standard_deviation
    double stddev = sqrt(posterior->posterior_for_points_only.covariance.data[0]);
    double spread = 3 + negative_observations.size();
    if( spread > 5 )
      spread = 5;
    for( size_t k = 0; (long)k < window.n; ++k ) {
      window.start.coordinate[k] -= spread * stddev;
      window.end.coordinate[k] += spread * stddev;
    }


    // debug
    // std::cout << "Points: ";
    // for( size_t i = 0; i < points.size(); ++i ) {
    //   std::cout << points[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Negative Obs: ";
    // for( size_t i = 0; i < negative_observations.size(); ++i ) {
    //   std::cout << negative_observations[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Covariance: " << to_eigen_mat(covariance) << std::endl;
    // std::cout << "Num ~ " << num_distribution << std::endl;
    // std::cout << "Prior ~ " << prior << std::endl;
    // std::cout << "Window: " << window << std::endl;
    // std::cout << "Posterior Poitns-Only ~ " << posterior->posterior_for_points_only << std::endl;
    // std::cout << "Posterior Scale: " << posterior->scale << std::endl;
    // std::cout << "Posterior Evals: " << std::endl;
    // for( double x = window.start.coordinate[0];
    // 	 x < window.end.coordinate[0];
    // 	 x += (window.end.coordinate[0] - window.start.coordinate[0]) / 1000.0 ) {
    //   std::cout << x << " " << (*posterior)( point(x) ) << std::endl;
    // }
    
    // now rejection sample from this posterior
    rejection_sampler_status_t status;
    nd_point_t m = 
      scaled_rejection_sample<nd_point_t>
      ( solid_function<nd_point_t,double>
	(boost::shared_ptr<math_function_t<nd_point_t,double> >(posterior)),
	posterior->scale,
	uniform_point_sampler_within_window( window ),
	status);

    // debug
    // std::cout << "Rejection Sampled Mixture Mean, iterations=" << status.iterations << "  " << status.seconds << " seconds" << std::endl;
    // std::cout << std::flush;

    return m;
  }


  //========================================================================
  
  // dense_matrix_t
  // resample_mixture_gaussian_covariance( const std::vector<nd_point_t> points,
  // 					const nd_point_t& mixture_mean,
  // 					const gamma_distribution_t& prior)
  // {

  //   size_t dim = 0;
  //   if( points.empty() == false ) {
  //     dim = points[0].n;
  //   }

  //   // compute the sum distance sqaured between points and mean
  //   double sum_distance = 0;
  //   for( size_t i = 0; i < points.size(); ++i ) {
  //     sum_distance += distance_sq( points[i], mixture_mean );
  //   }

  //   // create the new distributions
  //   gamma_distribution_t new_dist;
  //   new_dist.shape = prior.shape + points.size() / 2.0;
  //   //new_dist.rate = 1.0 / ( sum_distance + prior.shape * prior.rate );
  //   new_dist.rate = prior.rate + sum_distance / 2.0;
    
  //   // sample a new precision
  //   double prec = sample_from(new_dist);
    
  //   // returns a covariance matrix which is diagonal with given
  //   // precision for all elements
  //   return to_dense_mat( Eigen::MatrixXd::Identity( dim, dim ) * ( 1.0 / prec ) );
  // }
  
  //========================================================================
  
  dense_matrix_t
  resample_mixture_gaussian_covariance( const std::vector<nd_point_t>& points,
					const std::vector<nd_aabox_t>& negative_observations,
					const nd_point_t& mixture_mean,
					const gamma_distribution_t& prior)
  {

    size_t dim = mixture_mean.n;
    if( points.empty() == false ) {
      dim = points[0].n;
    }
    if( dim < 1 ) {
      throw std::domain_error("Cannot handle dimension of size < 1" );
    }

    // compute the sum distance sqaured between points and mean
    double sum_distance = 0;
    for( size_t i = 0; i < points.size(); ++i ) {
      sum_distance += distance_sq( points[i], mixture_mean );
    }

    // create the new distributions
    gamma_distribution_t new_dist;
    new_dist.shape = prior.shape + points.size() / 2.0;
    //new_dist.rate = 1.0 / ( sum_distance + prior.shape * prior.rate );
    new_dist.rate = prior.rate + sum_distance / 2.0;
    
    // sample a new precision
    double prec = sample_from(new_dist);
    if( prec == 0 )
      prec = 0.0000000001;
    

    // caclulate hte covariance
    double cov = 1.0 / prec;
    if( cov <= 1e-10 ) {
      cov = 1e-10;
    }
    if( isnan(cov) ) {
      // assume becasue of divide by 0
      cov = 1.0 / 1e-10; 
    }
    
    // returns a covariance matrix which is diagonal with given
    // precision for all elements
    return to_dense_mat( Eigen::MatrixXd::Identity( dim, dim ) * cov );
  }

  //========================================================================
  
  double
  resample_mixture_poisson_lambda( const std::vector<nd_point_t>& points,
				   const gamma_distribution_t& prior)
  {
    // new gamma
    gamma_distribution_t new_dist;
    new_dist.shape = prior.shape + points.size();
    new_dist.rate = prior.rate + 1;

    return sample_from( new_dist );
  }

  //========================================================================

  class alpha_posterior_likelihood_t
  {
  public:
    alpha_posterior_likelihood_t( double a,
				  double num_mix,
				  double num_ob )
      : alpha(a),
	num_mixtures( num_mix ),
	num_obs( num_ob )
    {}

    double alpha;
    double num_mixtures;
    double num_obs;
		  
    double operator()( const double& x ) const
    {
      return gsl_sf_gamma( x ) * pow(x, (double)num_mixtures - 2 ) * exp( - 1.0 / x ) / gsl_sf_gamma( x + num_obs );
    }
  };

  double 
  resample_alpha_hyperparameter( const double alpha,  
				 const std::size_t num_mixtures,
				 const std::size_t num_obs )
  {
     alpha_posterior_likelihood_t lik( alpha,
     				      num_mixtures,
     				      num_obs );
    // return rejection_sample<double>( lik, uniform_sampler_within_range(0.00000000001,num_obs) );

    // // Hack, just sample some values and sample from those
    // uniform_sampler_within_range uniform( 0.00001, 1 );
    // uniform_unsigned_long_sampler_within_range uniform_long( 1, num_obs + 2 );
    // std::vector<double> sample_alphas;
    // std::vector<double> sample_lik;
    // for( std::size_t i = 0; i < 1000; ++i ) {
    //   if( flip_coin( 0.5 ) ) {
    // 	double x = uniform();
    // 	sample_alphas.push_back( x );
    // 	sample_lik.push_back( lik(x) );
    //   } else {
    // 	double x = uniform_long();
    // 	sample_alphas.push_back( x );
    // 	sample_lik.push_back( lik(x) );
    //   }
    // }
    // discrete_distribution_t dist;
    // dist.n = sample_alphas.size();
    // dist.prob = sample_alphas;
    // return sample_alphas[sample_from(dist)];

     autoscaled_rejection_sampler_status_t status;
     boost::function1<double,double> lik_f = lik;
     double sampled_alpha =
       autoscale_rejection_sample<double>
       (lik_f, 0.00001, (double)num_obs + 2, status );
     return sampled_alpha;
  }

  //========================================================================

  gaussian_distribution_t
  _resample_mean_distribution_hyperparameters_for_dim
  ( igmm_point_process_state_t& state,
    int dim )
  {
    // assert( state.model.mean_distribution.dimension == 1 );
    // if( state.model.mean_distribution.dimension != 1 ) {
    //   throw std::domain_error( "Only implemented for 1D gaussians!" );
    // }
    
    double previous_precision = 1.0 / state.model.mean_distribution.covariance.data[state.model.mean_distribution.dimension * dim + dim];
    double hyperprior_precision = 1.0 / state.model.prior_variance;
    
    // ok, sum the means of the current mixtures
    double mean_sum = 0;
    for( std::size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
      mean_sum += state.mixture_gaussians[i].means[dim];
    }

    // compute the new variance of the distribution over means
    double new_variance = 
      1.0 / ( previous_precision * state.mixture_gaussians.size() + hyperprior_precision );
    
    // create the new gaussian for hte mean
    gaussian_distribution_t new_mean_dist;
    new_mean_dist.dimension = 1;
    new_mean_dist.means.push_back( ( previous_precision * mean_sum + state.model.prior_mean * hyperprior_precision ) * new_variance );
    new_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * new_variance );

    // sample a new mean
    nd_point_t new_mean = sample_from( new_mean_dist );
    
    // sum the sqaured ereror to this new mean
    // to compute distribution of new precision
    double sum_diff = 0;
    for( std::size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
      sum_diff += distance_sq( point(state.mixture_gaussians[i].means[dim]),
			       new_mean );
    }
    
    // crea the precision distribution
    gamma_distribution_t new_precision_dist;
    new_precision_dist.shape = ( state.mixture_gaussians.size() / 2.0 + state.model.precision_distribution.shape );
    //new_precision_dist.rate = 1.0 / ( 2 * ( sum_diff + 1.0/hyperprior_precision));
    new_precision_dist.rate = sum_diff / 2.0 + state.model.precision_distribution.rate;
    
    // sample a new precision
    double new_precision = sample_from( new_precision_dist );
    
    // return a new gaussian
    gaussian_distribution_t sampled_mean_dist;
    sampled_mean_dist.dimension = new_mean_dist.dimension;
    sampled_mean_dist.means = new_mean.coordinate;
    sampled_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * 1.0 / new_precision );

    return sampled_mean_dist;
    
  }

  //========================================================================

  gaussian_distribution_t
  resample_mean_distribution_hyperparameters
  ( igmm_point_process_state_t& state )
  {

    int dim = state.model.mean_distribution.dimension;

    // ok, we will treat each dimension of the gaussians as independant and
    // resample each individually
    std::vector<gaussian_distribution_t> individual_gaussians;
    for( size_t i = 0; i < dim; ++i ) {
      individual_gaussians.push_back( _resample_mean_distribution_hyperparameters_for_dim( state, i ) );
    }

    // now collect all individual gaussians into a single N-dim gaussian
    gaussian_distribution_t g;
    g.dimension = dim;
    for( auto ig : individual_gaussians ) {
      g.means.push_back( ig.means[0] );
    }
    g.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) );
    for( size_t i = 0; i < individual_gaussians.size(); ++i ) {
      g.covariance.data[ dim * i + i ] = individual_gaussians[i].covariance.data[0];
    }

    return g;
  }

  //========================================================================

  class precision_shape_posterior_t
  {
  public:
    double factor;
    double k;
    double rate;

    precision_shape_posterior_t( double factor, double k,
				  double rate )
      : factor(factor),
	k(k),
	rate(rate)
    {}

    double operator() (double b) const
    {
      double h = gsl_sf_gamma( b/2.0 );
      if( h > 1000 )
	return 0.0;
      if( h < 0.00001 )
	return 0.0;
      double r = 1.0 / pow( h, k );
      r *= pow( b * rate / 2.0, (k * b - 3.0) / 2.0 );
      r *= exp( - 1.0 / ( 2.0 * b ) );
      r *= factor;
      if( r > 10000 )
	return 0;
      return r;
    }
  };

  gamma_distribution_t
  resample_precision_distribution_hyperparameters( igmm_point_process_state_t& state )
  {
    // we are going to treat each dimension of each gaussian independently
    // and have the precision distribution boe over all of these

    int dim = state.model.mean_distribution.dimension;


    double b = state.model.precision_distribution.shape;
    double w = state.model.precision_distribution.rate;

    // sum the precisions of each mixture
    // as well as the factor for them
    double prec_sum = 0;
    double prec_factor = 1;
    for( std::size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
      for( int d = 0; d < dim; ++d ) {
	double prec = ( 1.0 / state.mixture_gaussians[i].covariance.data[dim * d + d] ); 
	prec_sum += prec;
	prec_factor *= pow( prec, b/2.0) * exp( - b * w * prec / 2.0 );
      }
    }

    // rejection sample from this a new shape
    // double new_shape
    //   = rejection_sample<double>
    //   ( precision_shape_posterior_t(prec_factor,
    // 				    state.mixture_gaussians.size(),
    // 				    state.model.precision_distribution.rate ),
    // 	uniform_sampler_within_range( 0.000001, 100 ) );

    precision_shape_posterior_t lik(prec_factor,
     				    state.mixture_gaussians.size() * dim,
     				    state.model.precision_distribution.rate );
      
    // Hack, just sample some values and sample from those
    uniform_sampler_within_range uniform( 0.00001, 100 );
    std::vector<double> sample_precs;
    std::vector<double> sample_vals;
    for( std::size_t i = 0; i < 100; ++i ) {
      double x = uniform();
      double l = lik(x);
      sample_vals.push_back( x );
      sample_precs.push_back( l );
    }
    discrete_distribution_t dist;
    dist.n = sample_precs.size();
    dist.prob = sample_precs;
    double new_shape = sample_vals[sample_from(dist)];


    // rejection_sampler_status_t status;
    // double low = 0.00001;
    // double high = 100;
    // while( lik(low) < 0.00001 ) {
    //   low *= 2.0;
    // }
    // while( lik(high) < 0.00001 ) {
    //   high /= 2.0;
    // }
    // while( lik(high) > lik(low) ) {
    //   high *= 2.0;
    // }
    
    // double new_shape
    //   = autoscale_rejection_sample<double>
    //   (lik, low, high, status );
    
    
    // now build up the distribution for the rate of the precision
    gamma_distribution_t new_rate_dist;
    new_rate_dist.shape = ( state.mixture_gaussians.size() * new_shape + 1 ) / 2.0;
    new_rate_dist.rate = 2 * 1.0 / ( new_shape * prec_sum + 1.0 / state.model.prior_variance );
    

    // sample a new precision rate
    double new_rate = sample_from(new_rate_dist);
    
    // return hte new distribution
    gamma_distribution_t new_dist;
    new_dist.shape = new_shape;
    new_dist.rate = new_rate;

    return new_dist;
  }

  //========================================================================

  gamma_distribution_t
  resample_num_points_per_gaussian_distribution_hyperparameters( igmm_point_process_state_t& state )
  {

    size_t sum_num = 0;
    for( size_t i = 0; i < state.mixture_gaussians.size(); ++i ) {
      sum_num += points_for_mixture( state, i ).size();
    }

    gamma_distribution_t new_dist;
    new_dist.shape = ( state.model.num_points_per_gaussian_distribution.shape + sum_num );
    new_dist.rate = state.model.num_points_per_gaussian_distribution.rate + state.mixture_gaussians.size();

    return new_dist;
  }

  //========================================================================

  // Descripiton:
  // Returns the probability that a negative region was seen for a 
  // particular mixture
  
  
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================


  void mcmc_single_step( igmm_point_process_state_t& state )
  {

    // For each observation, sample a corresponding mixture for it
    // from the known mixtures (with their parameters) as well as a
    // completely new mixture (using the DP-alpha parameter)

    for( size_t observation_i = 0; observation_i < state.observations.size(); ++observation_i ) {

      // remove mixture it this observation is the only observation in it
      size_t old_mixture_index = state.observation_to_mixture[ observation_i ];
      if( state.mixture_to_observation_indices[ state.observation_to_mixture[ observation_i ] ].size() == 1 ) {


	remove( state.mixture_gaussians, old_mixture_index );
	remove( state.mixture_poissons, old_mixture_index );
	remove( state.mixture_to_observation_indices, old_mixture_index );

	// fix obs->mixture mapping since we now have one less mixture
	for( size_t i = 0; i < state.observation_to_mixture.size(); ++i ) {
	  if( i == observation_i ) {
	    state.observation_to_mixture[i] = std::numeric_limits<size_t>::max();
	  } else if( state.observation_to_mixture[i] > old_mixture_index ) {
	    	    
	    state.observation_to_mixture[i] -= 1;
	  }
	}

	
	// set the old index to be the MAX since we just removed the
	// entire mixture (so ther eis no old mixture left)
	old_mixture_index = std::numeric_limits<size_t>::max();

      } 

      // compute the likelihood for each known cluster, and multiply
      // by the number of poitns in that cluster to get the likelihood
      // of the observation belonging to that cluster
      std::vector<double> likelihoods = std::vector<double>();
      for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {
	
	// number of point belonging to mixture (discounting ourselves)
	size_t num_obs_in_mixture = state.mixture_to_observation_indices[ mixture_i ].size();
	if( state.observation_to_mixture[ observation_i ] == mixture_i ) {
	  --num_obs_in_mixture;
	}

	// likelihood of this observation comming from this mixtiure
	double lik = 
	  pdf( state.observations[ observation_i ],
	       state.mixture_gaussians[ mixture_i ] )
	  * pdf( (unsigned int)num_obs_in_mixture + 1,
		 state.mixture_poissons[ mixture_i ] )
	  * ( num_obs_in_mixture / 
	      state.observations.size() - 1 + state.model.alpha );
	
	// store this likelihood (to later sample from it)
	likelihoods.push_back( lik );
      }

      // add the likelihood of a complete new mixture
      // here we use a SINGLE sample for the expected 
      // new model from our hyperparameters over both gauassians
      // as well as num points (the poisson)
      gaussian_distribution_t sampled_gaussian = sample_gaussian_from( state.model.mean_distribution,
								  state.model.precision_distribution );
      poisson_distribution_t sampled_poisson = sample_poisson_from( state.model.num_points_per_gaussian_distribution );
      double new_mixture_lik = 
	pdf( state.observations[ observation_i ],
	     sampled_gaussian )
	* pdf( 1, sampled_poisson )
	* ( state.model.alpha /
	    state.observations.size() - 1 + state.model.alpha );
      if( new_mixture_lik < 0 ) {
	new_mixture_lik = 0;
      }
      likelihoods.push_back( new_mixture_lik );

      // Now sample a mixture for this observation from the likelihoods
      size_t new_mixture_index = sample_from( discrete_distribution( likelihoods ) );
      
      // if this is a new mixture, sample new parameters for it
      if( new_mixture_index == likelihoods.size() - 1 ) {
	state.mixture_gaussians.push_back ( sample_gaussian_from( state.model.mean_distribution,
								  state.model.precision_distribution ) );
	state.mixture_poissons.push_back( sample_poisson_from( state.model.num_points_per_gaussian_distribution ) );
	state.mixture_to_observation_indices.push_back( std::vector<size_t>() );
      }
      
      // update correspondances and observation index mappings
      state.observation_to_mixture[ observation_i ] = new_mixture_index;
      state.mixture_to_observation_indices[ new_mixture_index ].push_back( observation_i );
      if( old_mixture_index < state.mixture_to_observation_indices.size() ) {
	for( size_t i = 0; i < state.mixture_to_observation_indices[ old_mixture_index ].size(); ++i ) {
	  if( state.mixture_to_observation_indices[ old_mixture_index ][ i ] == observation_i ) {
	    
	    remove( state.mixture_to_observation_indices[ old_mixture_index ],
		    i );
	    
	    break; 
	    // remove only the FIRST index (since we may have a double because
	    // we stayed at the same mixture
	  }
	}
      }

      // resample the mixture parameters
      for( size_t mixture_i = 0; mixture_i < state.mixture_gaussians.size(); ++mixture_i ) {

	// first get the points in the mixture component
	std::vector<nd_point_t> points = points_for_mixture( state, mixture_i );
	std::vector<nd_aabox_t> negative_observations = state.negative_observations;

	// Now resample the mixture number of points
	double mixture_num_lambda
	  = resample_mixture_poisson_lambda( points,
					     state.model.num_points_per_gaussian_distribution );
	
	// Ok, resmaple the mean of the mixture
	poisson_distribution_t num_dist;
	num_dist.lambda = mixture_num_lambda;
	nd_point_t mixture_mean 
	  = resample_mixture_gaussian_mean( points, 
					    negative_observations,
					    state.mixture_gaussians[mixture_i].covariance,
					    num_dist,
					    state.model.mean_distribution );
	
	// now resample the covariance of the mixture
	dense_matrix_t mixture_covariance
	  = resample_mixture_gaussian_covariance( points,
						  negative_observations,
						  mixture_mean,
						  state.model.precision_distribution );
	
	

	// set the new mixture parameters
	state.mixture_gaussians[ mixture_i ].means = mixture_mean.coordinate;
	state.mixture_gaussians[ mixture_i ].covariance = mixture_covariance;
	state.mixture_poissons[ mixture_i ].lambda = mixture_num_lambda;
	
      }


      // // ok, compute posteriors of the hyperparameter distributions

      // Resample a new alpha
      state.model.alpha 
	= resample_alpha_hyperparameter( state.model.alpha,
					 state.mixture_gaussians.size(),
					 state.observations.size() );
      
      // Resample new mean distribution
      state.model.mean_distribution
	= resample_mean_distribution_hyperparameters( state );
      
      // resample the new precision distribution
      state.model.precision_distribution
	= resample_precision_distribution_hyperparameters( state );
      
      // resample the number of poitns distribution
      state.model.num_points_per_gaussian_distribution
	= resample_num_points_per_gaussian_distribution_hyperparameters( state );     

    }
    
  }
			 

  //========================================================================


  // BAD not working code inside MCMC
	// // calculate mean
	// nd_point_t obs_mean_mean = mean( points_for_mixture( state, mixture_i ) );
	// double obs_mean_variance = variance( points_for_mixture( state, mixture_i ) );

	// gaussian_distribution_t mean_gaussian;
	// mean_gaussian.dimension = obs_mean_mean.n;
	// for( size_t i = 0; i < obs_mean_mean.n; ++i ) {
	//   mean_gaussian.means.push_back( obs_mean_mean.coordinate[i] );
	// }
	// mean_gaussian.covariance = to_dense_mat( Eigen::MatrixXd::Identity( mean_gaussian.dimension, mean_gaussian.dimension ) * obs_mean_variance );
	// state.mixture_gaussians[ mixture_i ].means = sample_from( mean_gaussian ).coordinate;
	// state.model.mean_distribution = mean_gaussian;
	// size_t dim = state.mixture_gaussians[ mixture_i ].dimension;
	
	// // calculate variance
	// double obs_variance_shape = ( state.model.precision_distribution.shape + state.mixture_to_observation_indices[ mixture_i ].size() ) / 2.0;
	// double obs_variance_rate = 1.0 / ( 2 * obs_mean_variance + state.model.precision_distribution.shape * state.model.precision_distribution.rate );
	// gamma_distribution_t variance_gamma;
	// variance_gamma.shape = obs_variance_shape;
	// variance_gamma.rate = obs_variance_rate;
	// state.model.precision_distribution = variance_gamma;
	// state.mixture_gaussians[ mixture_i ].covariance = to_dense_mat( Eigen::MatrixXd::Identity( dim, dim ) * 1.0 / sample_from( variance_gamma ) );
	
	// // calculate gamma parameters for the number of points
	// // per gauassian (a poisson )
	// double num_shape =  state.model.num_points_per_gaussian_distribution.shape + state.mixture_to_observation_indices[ mixture_i ].size();
	// double num_rate = 1.0 / state.model.num_points_per_gaussian_distribution.rate + 1;
	// gamma_distribution_t lambda_gamma;
	// lambda_gamma.shape = num_shape;
	// lambda_gamma.rate = num_rate;
	// state.model.num_points_per_gaussian_distribution = lambda_gamma;
	// state.mixture_poissons[ mixture_i ].lambda = sample_from( lambda_gamma );


  //========================================================================

       // // draw new alpha
      // state.model.alpha = sample_new_alpha( state.mixture_gaussians.size(),
      // 					    state.observations.size() );
      
      // // update the mean distribution
      // nd_point_t mean_mean_sample = point(state.model.mean_distribution.means);
      // double mean_variance_sample = state.model.mean_distribution.covariance.data[0];
      // nd_point_t mean_mean_posterior = point( mean_mean_sample.n );
      // for( size_t i = 0; i < mean_mean_posterior.n; ++i ) {
      // 	double c = 0;
      // 	for( size_t j = 0; j < state.mixture_gaussians.size(); ++j ) {
      // 	  c += state.mixture_gaussians[j].means[i];
      // 	}
      // 	c *= (1.0 / mean_variance_sample);
      // 	c += state.model.prior_mean / state.model.prior_variance;
      // 	c *= ( 1.0 / ( (1.0/mean_variance_sample) * state.mixture_gaussians.size() + (1.0/state.model.prior_variance)));
      // 	mean_mean_posterior.coordinate[i] = c;
      // }
      // dobule mean_variance_posterior = ( 1.0 / ( (1.0/mean_variance_sample) * state.mixture_gaussians.size() + (1.0/state.model.prior_variance)));
      // size_t nd = mean_mean_posterior.n;
      // gaussian_distribution_t mean_posterior;
      // mean_posterior.means = mean_mean_posterior.coordinate;
      // mean_posterior.covariance = to_dense_mat( Eigen::MatrixXd::Identity(nd,nd) * mean_variance_posterior );
   
      // // sample the mean of mean distribution
      // state.model.mean_distribution = sample_from( mean_posterior ).coordinate;
      
      // // sample the varuance of mean distribution
 

  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================
  //========================================================================



}
