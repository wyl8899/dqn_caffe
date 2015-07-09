#ifndef __DQN_SOLVER_H__
#define __DQN_SOLVER_H__

#include "CommonIncludes.h"
#include "State.h"
#include "Environment.h"
#include "Transition.h"
#include "ExpHistory.h"

#include "caffe/util/upgrade_proto.hpp"

DECLARE_double(gamma);
DECLARE_double(epsilon);
DECLARE_int32(learn_start);
DECLARE_int32(history_size);
DECLARE_int32(update_freq);
DECLARE_int32(frame_skip);
DECLARE_int32(eval_episodes);
DECLARE_int32(eval_freq);
DECLARE_int32(sync_freq);
DECLARE_int32(normalize);

using caffe::SolverState;
using caffe::NetParameter;

template <typename Dtype>
class DqnSolver : public SGDSolver<Dtype> {
public:
  explicit DqnSolver( const SolverParameter& param );
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
  
  void Evaluate();
  
protected:
  // predefined hyperparameters
  float gamma_;     // discount factor
  float epsilon_;   // exploration used in evaluation
  int learnStart_;  // iterations before learning starts
  int updateFreq_;  // actions between successive update
  int frameSkip_;   // frames between successive actions
  int evalFreq_;    // iterations between evaluations
  int syncFreq_;    // iterations between target_net sync

  ExpHistory<Dtype> expHistory_;
  Environment<Dtype> environment_;
  
  // a separate net, termed target_net, is used to calculate target update value
  // the target_net is synced with the training net every syncFreq_ iterations.
  shared_ptr<Net<Dtype> > targetNet_;
  
  // cached information and pointers  
  int lossLayerID_;
  
  typedef Blob<Dtype>* pBlob;
  pBlob stateBlob_, rewardBlob_, predBlob_, actionBlob_;
  pBlob stateTargetBlob_, actionTargetBlob_;
  
  int legalActionCount_;
  
  // running average of (dL / dw)^2 and tmp for calculation
  vector<shared_ptr<Blob<Dtype> > > sqGrad_, tmpGrad_;
  
  void InitTargetNet();
  void SyncTargetNet();
  
  void FeedReward( int action, float reward );
  
  virtual float GetEpsilon();
  int GetActionFromNet();
  inline int GetRandomAction() {
    return rand() % legalActionCount_;
  }
  int GetAction( float epsilon );
  
  void Step( int );
  State<Dtype> PlayStep( State<Dtype> state, float* totalReward, float epsilon );
  void TrainStep();
  
  void ZeroGradients();
  void ApplyUpdate();
  void ComputeUpdateValue( int param_id, Dtype rate );
  
  virtual void SnapshotSolverState( SolverState* state );
  virtual void RestoreSolverState( const SolverState & state );
};

#endif // __DQN_SOLVER_H__
