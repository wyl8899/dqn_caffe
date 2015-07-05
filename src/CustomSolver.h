#ifndef __CUSTOM_SOLVER_H__
#define __CUSTOM_SOLVER_H__

#include "CommonIncludes.h"
#include "State.h"
#include "Environment.h"
#include "Transition.h"
#include "ExpHistory.h"

DECLARE_double(gamma);
DECLARE_double(epsilon);
DECLARE_int32(learn_start);
DECLARE_int32(history_size);
DECLARE_int32(update_freq);
DECLARE_int32(frame_skip);

using caffe::SolverState;

template <typename Dtype>
class CustomSolver : public SGDSolver<Dtype> {
public:
  explicit CustomSolver( const SolverParameter& param );
  void Solve( const char* resume_file = NULL );
  inline void Solve( const string resume_file ) {
    Solve( resume_file.c_str() ); 
  }
  inline void InitializeALE( ALEInterface* ale ) {
    environment_.SetALE( ale );
    legalActionCount_ = environment_.GetLegalActionCount();
    CHECK_EQ( legalActionCount_, predBlob_->count() )
      << "Size of output blob \"pred\" should equal the number of "
         "legal actions";
    CHECK_EQ( legalActionCount_, rewardBlob_->count() )
      << "Size of input blob \"reward\" should equal the number of "
         "legal actions";
  }
protected:  
  // predefined hyperparameters
  float gamma_;     // discount factor
  float epsilon_;   // exploration used in evaluation
  int learnStart_;  // iterations before learning starts
  int updateFreq_;  // actions between successive update
  int frameSkip_;   // frames between successive actions

  ExpHistory<Dtype> expHistory_;
  Environment<Dtype> environment_;
  
  // cached information and pointers  
  int lossLayerID_;
  Blob<Dtype> *stateBlob_;
  Blob<Dtype> *rewardBlob_;
  Blob<Dtype> *predBlob_;
  Blob<Dtype> *actionBlob_;
  
  int legalActionCount_;
  
  // running average of (dL / dw)^2 and tmp for calculation
  vector<shared_ptr<Blob<Dtype> > > sqGrad_, tmpGrad_;
  
  void FeedState();
  void FeedReward( int action, float reward );
  
  virtual float GetEpsilon();
  int GetActionFromNet();
  inline int GetRandomAction() {
    return rand() % legalActionCount_;
  }
  int GetAction();
  
  void Step( int );
  State<Dtype> PlayStep( State<Dtype> state, float & totalReward );
  Dtype TrainStep();
  
  void ZeroGradients();
  void ApplyUpdate();
  void ComputeUpdateValue( int param_id, Dtype rate );
  
  virtual void SnapshotSolverState( SolverState* state );
  virtual void RestoreSolverState( const SolverState & state );
};

#endif // __CUSTOM_SOLVER_H__
