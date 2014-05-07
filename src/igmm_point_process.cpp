#include "igmm_point_process.hpp"
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>


namespace igmm_point_process {

  static double _dg_1 = boost::math::digamma( 1.0 );
  static double _dg_2 = boost::math::digamma( 2.0 );


  double
  igmm_point_process_t::expected_posterior_entropy_difference
  ( const std::vector<math_core::nd_point_t>& new_obs ) const
  {
    double N = new_obs.size();
    double a = this->_state.model.alpha;
    double sum = 0;
    for( size_t k = 0; k < new_obs.size(); ++k ) {
      sum += ( 1.0 / ( a + 1 + k ) );
    }
    //std::cout << "  igmm:ent-diff  " << ( ( ( a / (a + N) ) - 1.0 ) * _dg_1 ) << "  " << ( ( N / (a + N) ) * _dg_2 ) << "  - " << sum << std::endl;
    return fabs( ( ( ( a / (a + N) ) - 1.0 ) * _dg_1 )
		 + ( ( N / (a + N) ) * _dg_2 )
		 - sum );
  }

}
