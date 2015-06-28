#ifndef __COMMON_INCLUDES_H__
#define __COMMON_INCLUDES_H__

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"

using std::string;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

using caffe::SGDSolver;
using caffe::SolverParameter;
using caffe::caffe_set;

#endif // __COMMON_INCLUDES_H__
