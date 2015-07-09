#ifndef __EXP_HISTORY_H__
#define __EXP_HISTORY_H__

#include "CommonIncludes.h"
#include "Transition.h"

template <typename Dtype>
class ExpHistory {
public:
  ExpHistory ( int capacity );
  
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
ExpHistory<Dtype>::ExpHistory ( int capacity ) : capacity_( capacity ) {
}

template <typename Dtype>
void ExpHistory<Dtype>::AddExperience( Transition<Dtype> exp ) {
  int cur;
  if ( size() < capacity_ ) {
    cur = size();
    history_.push_back( exp );
  } else {
    cur = currentIndex_ + 1;
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
