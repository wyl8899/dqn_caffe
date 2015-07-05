#ifndef __CUSTOM_TEST_SOLVER_H__
#define __CUSTOM_TEST_SOLVER_H__

#include "CommonIncludes.h"
#include "CustomSolver.h"

template <typename Dtype>
class CustomTestSolver : public CustomSolver<Dtype> {
public:
  explicit CustomTestSolver( const SolverParameter& param )
    : CustomSolver<Dtype>( param ) {}
  void Solve( const char* resume_file = NULL );
  inline void Solve( const string resume_file ) {
    Solve( resume_file.c_str() ); 
  }
protected:
  inline virtual float GetEpsilon() {
    return this->epsilon_;
  }
  State<Dtype> PlayStep( State<Dtype> nowState, float & totalReward );
};

#endif // __CUSTOM_TEST_SOLVER_H__
