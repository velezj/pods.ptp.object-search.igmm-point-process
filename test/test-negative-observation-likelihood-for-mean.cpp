
#include <igmm-point-process/mcmc.hpp>
#include <math-core/matrix.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace math_core;
using namespace probability_core;
using namespace igmm_point_process;


static std::string colors = "rgbcmyk";


void create_matlab_file_for_posterior
( const std::string& filename,
  const std::vector<nd_point_t> & points,
  const std::vector<nd_aabox_t>& negative_observations,
  const dense_matrix_t& covariance,
  const poisson_distribution_t& num_distribution,
  const gaussian_distribution_t& prior,
  const unsigned long num_eval_samples,
  const double eval_spread,
  const long index )
{
  
  Eigen::MatrixXd inverse_covariance = to_eigen_mat( covariance ).inverse();

  // create a new posterior function
  boost::shared_ptr<gaussian_mixture_mean_posterior_t>
    posterior( new gaussian_mixture_mean_posterior_t 
	       ( points,
		 negative_observations,
		 inverse_covariance,
		 num_distribution,
		 prior ) );
  
  // creat the legend/display name for this plot
  std::ostringstream oss;
  oss << points.size() << "_{ptr}" << negative_observations.size() << "_{nr}"
      << covariance.data[0] << "_{cov}" << " " << num_distribution 
      << " " << prior;
  
  // open the ifle
  std::ofstream fout( filename.c_str() );
  
  // ok, calculate the range given the spread
  // as well as the initial x and step size
  double points_only_sigma = sqrt( posterior->posterior_for_points_only.covariance.data[0]);
  double min_x = posterior->posterior_for_points_only.means[0] - eval_spread * points_only_sigma;
  double max_x = posterior->posterior_for_points_only.means[0] + eval_spread * points_only_sigma;
  double step_x = (max_x - min_x) / num_eval_samples;

  // ok, take samples at step_x intervals and create a matlab script for it
  fout << "data_x = [ ";
  for( double x = min_x; x <= max_x; x += step_x ) {
    fout << x << " ";
  }
  fout << "];" << std::endl;
  fout << "data_px = [ ";
  for( double x = min_x; x <= max_x; x += step_x ) {
    fout << (*posterior)( point(x) ) << " ";
  }
  fout << "];" << std::endl;
  
  // add the plotting and labeling to script
  fout << "m = sum( (data_px./sum(data_px)) .* data_x );" << std::endl;
  fout << "h = plot([m m],[0 max(data_px)/sum(data_px)], '" << colors[index % colors.size()] << "--');" << std::endl;
  fout << "set(h,{'DisplayName'},{'mean'});" << std::endl;
  fout << "h = plot(data_x,data_px ./ sum(data_px), '" << colors[index % colors.size()] << "-' );" << std::endl;
  fout << "set(h,{'DisplayName'},{'" << oss.str() << "'});" << std::endl;
  
  fout.close();
}


int main()
{

  // some points
  std::vector<nd_point_t> points;
  points.push_back( point(0.0) );
  points.push_back( point(0.5) );
  points.push_back( point(-0.5) );
  
  // some negaative obs
  std::vector<nd_aabox_t> negative_observations;
  

  // covariance
  dense_matrix_t covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * 1 );

  // distributions of points
  poisson_distribution_t num_distribution;
  num_distribution.lambda = 30;
  
  // the prior
  gaussian_distribution_t prior;
  prior.dimension = 1;
  prior.means.push_back( 0.0 );
  prior.covariance = covariance;


  // ok, plot with different number of negative observations
  double region_size = 1;
  long index = 0;
  for( double no_x = -3 * covariance.data[0]; no_x < 12 * covariance.data[0]; no_x += region_size, ++index )  {
    std::ostringstream oss;
    oss << "plot_" << negative_observations.size() << ".m";
    create_matlab_file_for_posterior
      ( oss.str(),
	points,
	negative_observations,
	covariance,
	num_distribution,
	prior,
	10000,
	10,
	index);
    nd_aabox_t nr;
    nr.n = 1;
    nr.start = point( no_x );
    nr.end = point( no_x + region_size );
    negative_observations.push_back( nr );
  }
  
  
  return 0;
}
