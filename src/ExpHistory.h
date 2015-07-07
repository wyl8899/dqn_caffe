#ifndef __EXP_HISTORY_H__
#define __EXP_HISTORY_H__

#include "CommonIncludes.h"
#include "Transition.h"

template <typename Dtype>
class ExpHistory {
public:
  ExpHistory ( int n );
  
  void AddExperience( Transition<Dtype> exp );
  Transition<Dtype> & Sample();
  inline size_t size() {
    return history_.size();
  }
  
private:
  vector<Transition<Dtype> > history_;
  int capacity_, currentIndex_;
};

template <typename Dtype>
ExpHistory<Dtype>::ExpHistory ( int n ) : capacity_( n ) {
}

template <typename Dtype>
void ExpHistory<Dtype>::AddExperience( Transition<Dtype> exp ) {
  int cur;
  if ( history_.size() < capacity_ ) {
    history_.push_back( exp );
    cur = history_.size();
  } else {
    cur = currentIndex_++;
    if ( cur == capacity_ )
      cur = 0;
    history_[cur] = exp;
  }
  currentIndex_ = cur;
}

template <typename Dtype>
Transition<Dtype> & ExpHistory<Dtype>::Sample() {
  return history_[rand() % history_.size()];
}

#endif // __EXP_HISTORY_H__
