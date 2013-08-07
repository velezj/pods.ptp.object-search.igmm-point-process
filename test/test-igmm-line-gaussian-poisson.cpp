
#include <point-process-experiment-core/simulated_data.hpp>
#include <math-core/io.hpp>
#include <math-core/matrix.hpp>
#include <igmm-point-process/mcmc.hpp>
#include <probability-core-graphics/lcmgl_distributions.hpp>
#include <iostream>

using namespace math_core;
using namespace probability_core;
using namespace probability_core::graphics;
using namespace point_process_experiment_core;
using namespace igmm_point_process;


void colorize( long i, double& r, double& g, double& b )
{
  static long color_n = 11;
  static double color_r[] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
  static double color_g[] = { 0.3, 0.2, 0.1, 0.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4 };
  static double color_b[] = { 0.2, 0.3, 0.5, 0.7, 0.2, 0.3, 0.5, 0.7, 0.2, 0.3, 0.5 };
  
  
  r = color_r[ i % color_n ];
  g = color_g[ i % color_n ];
  b = color_b[ i % color_n ];
}

int main( int argc, char** argv )
{

  lcm_t* lcm = lcm_create(NULL);
  bot_lcmgl_t* igmm_lcmgl = bot_lcmgl_init( lcm, "IGMM" );
  bot_lcmgl_t* points_lcmgl = bot_lcmgl_init( lcm, "IGMM-TRUE-POINTS" );
  bot_lcmgl_t* lcmgl;

  // craete the window (the range)
  nd_aabox_t window;
  window.n = 1;
  window.start = point( 0 );
  window.end = point( 500 );

  // simulate some points
  std::vector<nd_point_t> points 
    = simulate_line_point_clusters_gaussian_spread_poisson_size
    ( window,
      5,
      2,
      10.0 );

  // draw the tru points
  lcmgl = points_lcmgl;
  lcmglColor3f( 0,0,0  );
  lcmglPushMatrix();
  lcmglScalef( 0.1, 1, 1 );
  lcmglPointSize( 3.0 );
  double point_radius = 0.1;
  lcmglBegin( LCMGL_POINTS );
  for( size_t i = 0; i < points.size(); ++i ) {
    double loc[] = { points[i].coordinate[0], 0, 0 }; 
    lcmglVertex2d( points[i].coordinate[0], 0 );
  }
  lcmglEnd();
  lcmglPopMatrix();
  lcmglPointSize( 1.0 );
  bot_lcmgl_switch_buffer( points_lcmgl );
  
  
  // now create a new point process;
  igmm_point_process_model_t model;
  model.alpha = 1;
  model.mean_distribution.dimension = 1;
  model.mean_distribution.means.push_back( 50 );
  model.mean_distribution.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * 1.0 );
  model.precision_distribution.shape = 2;
  model.precision_distribution.rate = 0.25;
  model.num_points_per_gaussian_distribution.shape = 2;
  model.num_points_per_gaussian_distribution.rate = 0.25;
  model.prior_mean = 0;
  model.prior_variance = 4;
  
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


  // sample for a while (burnin)
  long num = 10000;
  for( size_t i = 0; i < num; ++i ) {
    mcmc_single_step( state );
    
    if( i % (num / 10) == 0 ) {
      
      // draw the mixtures
      lcmgl = igmm_lcmgl;
      double r,g,b;
      double zero_p[] = { 0,0,0 };
      for( std::size_t mi = 0; mi < state.mixture_gaussians.size(); ++mi ) {
	colorize( mi, r,g,b );
	lcmglPushMatrix();
	lcmglScalef( 0.1, 10.0, 1.0 );
	lcmglColor3f( r, g, b );
	draw_distribution( lcmgl, state.mixture_gaussians[mi] );
	lcmglColor3f( 1, 1, 1 );
	bot_lcmgl_text( lcmgl, zero_p, "Gauss" );
	lcmglPopMatrix();
	lcmglPushMatrix();
	lcmglTranslated( 0.0, 10.0, 0.0 );
	lcmglScalef( 0.1, 10.0, 1.0 );
	lcmglTranslated( state.mixture_gaussians[mi].means[0], 0.0, 0.0 );
	lcmglColor3f( r, g, b );
	draw_distribution( lcmgl, state.mixture_poissons[mi] );
	lcmglColor3f( 1,1,1 );
	bot_lcmgl_text( lcmgl, zero_p, "Poss" );
	lcmglPopMatrix();
      }
      lcmglPushMatrix();
      lcmglTranslated( 0, 20, 0 );
      lcmglScalef( 0.1, 10, 1 );
      lcmglColor3f( 0,0,0 );
      draw_distribution( lcmgl, state.model.mean_distribution );
      lcmglColor3f( 1,1,1 );
      bot_lcmgl_text( lcmgl, zero_p, "Mean" );
      lcmglPopMatrix();
      lcmglTranslated( 0, 30, 0 );
      lcmglScalef( 0.1, 10, 1 );
      lcmglColor3f( 0,0,0 );
      draw_distribution( lcmgl, state.model.precision_distribution );
      lcmglColor3f( 1,1,1 );
      bot_lcmgl_text( lcmgl, zero_p, "Prec" );
      lcmglPopMatrix();
      lcmglTranslated( 0, 40, 0 );
      lcmglScalef( 0.1, 10, 1 );
      lcmglColor3f( 0,0,0 );
      draw_distribution( lcmgl, state.model.num_points_per_gaussian_distribution );
      lcmglColor3f( 1,1,1 );
      bot_lcmgl_text( lcmgl, zero_p, "Num" );
      lcmglPopMatrix();
      bot_lcmgl_switch_buffer( igmm_lcmgl );
      
      std::cout << "[" << i << "]--------------------------------------------" << std::endl;
      std::cout << state << std::endl;
      std::cout << "--------------------------------------------" << std::endl;
    }

    //lcm_handle( lcm );
  }

  std::cout << state << std::endl;

  return 0;

}

