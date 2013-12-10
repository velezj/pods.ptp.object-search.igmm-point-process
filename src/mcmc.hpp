
#if !defined( __IGMM_POINT_PROCESS_MCMC_HPP__ )
#define __IGMM_POINT_PROCESS_MCMC_HPP__

#include "model.hpp"
#include <math-core/math_function.hpp>
#include <math-core/matrix.hpp>
#include <math-core/io.hpp>
#include <probability-core/distribution_utils.hpp>
#include <gsl/gsl_sf_erf.h>
#include <iostream>
#include <stdexcept>


namespace igmm_point_process {

  using namespace math_core;
  using namespace probability_core;

  

  // Description:
  // Runs mcmc for a given number of iterations.
  // This WILL change the given state
  void run_mcmc( igmm_point_process_state_t& state, 
		 const size_t num_iterations,
		 bool );


  // Description:
  // Perform a single step of mcmc.
  // This WILL change the given state
  void mcmc_single_step( igmm_point_process_state_t& state );

  





  
  // Description:
  // The likelihood of a negtaive observation (a region) given
  // a covariance ( so the mean is the function input / domain )
  class negative_observation_likelihood_for_mean_t
    : public math_function_t<nd_point_t,double>
  {
  public:
    nd_aabox_t region;
    dense_matrix_t covariance;
    double num_points_lambda;
    negative_observation_likelihood_for_mean_t
    ( const nd_aabox_t& region,
      const dense_matrix_t& covariance,
      const double& num_points_lambda )
      : region(region),
	covariance( covariance ),
	num_points_lambda( num_points_lambda )
    {}

    virtual ~negative_observation_likelihood_for_mean_t() {}
    
    virtual
    double operator() ( const nd_point_t& mu ) const
    {
      // treat each dimension of mean independently
      double p = 1;
      for( size_t i = 0; (long)i < mu.n; ++i ) {
	double x = mu.coordinate[i];
	double a = region.start.coordinate[i];
	double b = region.end.coordinate[i];
	double sig = to_eigen_mat(covariance)(i,i);
	double amass = gsl_sf_erfc( - ( a - x ) / ( sqrt(2.0) * sig ) );
	double bmass = gsl_sf_erfc( - ( b - x ) / ( sqrt(2.0) * sig ) );
	double diff = amass - bmass;
	double lik = exp( 0.5 * num_points_lambda * diff );
	p *= lik;
	
	if( p < 1.0e-20 ) {
	  std::cout << "  -- neg_obs_lik very small: " << p << " at: " << mu << std::endl;
	}
      }
      return p;
    }
  };


  // Descripiton:
  // The posterior distribution of a cluster  mean given both adata points
  // and negative observations (no longer conjugate hence we 
  // need the explicit posterior function )
  class gaussian_mixture_mean_posterior_t
    : public math_function_t<nd_point_t,double>
  {
  public:
    std::vector<nd_point_t> points;
    std::vector<nd_aabox_t> negative_observations;
    dense_matrix_t covariance;
    poisson_distribution_t num_distribution;
    gaussian_distribution_t prior;
    gaussian_distribution_t posterior_for_points_only;
    boost::shared_ptr<math_function_t<nd_point_t,double> > posterior;

    double scale;
    
    gaussian_mixture_mean_posterior_t
    ( const std::vector<nd_point_t>& points,
      const std::vector<nd_aabox_t>& negative_observations,
      const dense_matrix_t& cov,
      const poisson_distribution_t& num_distribution,
      const gaussian_distribution_t& prior )
      : points(points),
	negative_observations(negative_observations),
	covariance( cov ),
	num_distribution( num_distribution ),
	prior( prior )
    {
      calculate_posterior();
    }

    void calculate_points_only_posterior()
    {
      // the dimension of the poitns
      std::size_t dim = 0;
      if( points.empty() == false )
	dim = points[0].n;
      else
	throw std::runtime_error( "no points for posterior!" );
      
      // invert the covariance to get a precision
      // (Not sure if this actually works for anything greater than 1D poitns!!)
      Eigen::MatrixXd prec_mat = to_eigen_mat( covariance ).inverse();
      
      // calculate the sum of the data points
      Eigen::VectorXd sum_vec( dim );
      for( size_t i = 0; i < dim; ++i ) {
	sum_vec(i) = 0;
      }
      for( size_t i = 0; i < points.size(); ++i ) {
	for( size_t k = 0; k < dim; ++k ) {
	  sum_vec(k) += points[i].coordinate[k];
	}
      }
      
      
      // Invert the prior's covariance to get a prior precision
      // and get the prior mean as Eign vector
      Eigen::MatrixXd prior_prec = to_eigen_mat( prior.covariance ).inverse();
      Eigen::VectorXd prior_mean = to_eigen_mat( prior.means );
      
      // Ok, now calculate the distribution over the new mixture mean
      Eigen::MatrixXd new_dist_prec = prec_mat * points.size() + prior_prec;
      Eigen::MatrixXd new_dist_cov = new_dist_prec.inverse();
      Eigen::VectorXd new_dist_mean 
	= new_dist_cov * ( prec_mat * sum_vec + prior_prec * prior_mean );
      
      // This is the posterior if you only include the observation points
      // and do NOT use the negative observations (so conjugate hence gaussian)
      gaussian_distribution_t new_dist;
      new_dist.dimension = dim;
      new_dist.means = to_vector( new_dist_mean ).component;
      new_dist.covariance = to_dense_mat( new_dist_cov );

      
      if( new_dist.means.size() > 1 &&
	  new_dist.means[1] > 1000 ) {

	std::cout << "posterior (points only) = " << new_dist << std::endl;
	std::cout << "   -- cov: " << to_eigen_mat( covariance ) << std::endl;
	std::cout << "   -- prec_mat: " << prec_mat << std::endl;
	std::cout << "   -- new_dist_mean: " << new_dist_mean << std::endl;
	std::cout << "   -- sum_vec: " << sum_vec << std::endl;
	std::cout << "   -- prior_prec: " << prior_prec << std::endl;
	std::cout << "   -- prior_mean: " << prior_mean << std::endl;
      }

      posterior_for_points_only = new_dist;
    }

    void calculate_posterior()
    {

      // cacluate the posterior using only the points and nopt
      // the negative regions
      calculate_points_only_posterior();

      // create the posterior math function 
      // (we will start with the point posterior and multiply by the likelihood
      // of the negaztive observatiosn )
      posterior = functions::gaussian_pdf( posterior_for_points_only );

      // Now, we need to multiply by the probability of *each* negative 
      // observation
      for( size_t i = 0; i < negative_observations.size(); ++i ) {
	boost::shared_ptr<math_function_t<nd_point_t,double> > neg_lik( new negative_observation_likelihood_for_mean_t( negative_observations[i], covariance, num_distribution.lambda ) );
	//posterior = neg_lik * posterior;
      }

      // set the scale to the mean with only the points
      this->scale = pdf( point(posterior_for_points_only.means),
			 posterior_for_points_only );
    }

    double operator() ( const nd_point_t& mu ) const
    {
      return (*posterior)(mu);
    }
    
  };

  
}

#endif

