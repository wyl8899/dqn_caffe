#ifndef __CUSTOM_SOLVER_H__
#define __CUSTOM_SOLVER_H__

#include "CommonIncludes.h"
#include "State.h"
#include "Environment.h"
#include "Transition.h"
#include "ExpHistory.h"

using caffe::SolverState;

template <typename Dtype>
class CustomSolver : public SGDSolver<Dtype> {
public:
  explicit CustomSolver( const SolverParameter& param );
  void Solve( const char* resume_file = NULL );
  inline void Solve( const string resume_file ) {
    Solve( resume_file.c_str() ); 
  }
  inline void SetALE( ALEInterface* ale ) {
    environment_.SetALE( ale );
    legalActionCount_ = environment_.GetLegalActionCount();
    CHECK_EQ( legalActionCount_, predBlob_->count() )
      << "Output size of the network should equal the number of "
         "legal actions";
  }
protected:
  enum {
    REPLAY_START_SIZE = 100,
    HISTORY_SIZE = 500000,
    UPDATE_FREQUENCY = 4,
    FRAME_SKIP = 3
  };
  
  ExpHistory<Dtype> expHistory_;
  Environment<Dtype> environment_;
  
  int lossLayerID_;
  Blob<Dtype> *stateBlob_;
  Blob<Dtype> *rewardBlob_;
  Blob<Dtype> *predBlob_;
  Blob<Dtype> *actionBlob_;
  
  int legalActionCount_;
  
  vector<shared_ptr<Blob<Dtype> > > sqGrad_, tmpGrad_;
  
  void FeedState();
  void FeedReward( int action, float reward );
  
  float GetEpsilon();
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
