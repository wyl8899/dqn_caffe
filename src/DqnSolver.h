#ifndef __DQN_SOLVER_H__
#define __DQN_SOLVER_H__

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
DECLARE_int32(eval_episodes);
DECLARE_int32(eval_freq);

using caffe::SolverState;

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
  
  void Evaluate() {
    const int count = FLAGS_eval_episodes;
    float reward = 0.0;
    State<Dtype> state;
    for ( int i = 0; i < count; ++i ) {
      environment_.ResetGame();
      state = environment_.GetState( true );
      while ( state.isValid() ) {
        state = PlayStep( state, &reward, epsilon_ );
      }
      // reward is accumulated to calculate average directly
    }
    LOG(INFO) << "Evaluate: Average score = " << reward / count
      << " over " << count << " game(s).";
  }
  
protected:
  // predefined hyperparameters
  float gamma_;     // discount factor
  float epsilon_;   // exploration used in evaluation
  int learnStart_;  // iterations before learning starts
  int updateFreq_;  // actions between successive update
  int frameSkip_;   // frames between successive actions
  int evalFreq_;    // iterations between evaluations

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
  int GetAction( float epsilon );
  
  void Step( int );
  State<Dtype> PlayStep( State<Dtype> state, float* totalReward, float epsilon );
  Dtype TrainStep();
  
  void ZeroGradients();
  void ApplyUpdate();
  void ComputeUpdateValue( int param_id, Dtype rate );
  
  virtual void SnapshotSolverState( SolverState* state );
  virtual void RestoreSolverState( const SolverState & state );
};

#endif // __DQN_SOLVER_H__
