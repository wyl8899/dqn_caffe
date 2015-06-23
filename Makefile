PROJECT := caffe
EXEC := caffe

CAFFE_ROOT := /home/wyl8899/projects/caffe
CAFFE_INCLUDE := $(CAFFE_ROOT)/include
CAFFE_LIB := $(CAFFE_ROOT)/build/lib
INCLUDE_DIRS += $(CAFFE_INCLUDE)
INCLUDE_DIRS += $(CAFFE_ROOT)/build/src
LIBRARY_DIRS += $(CAFFE_LIB)

BLAS_INCLUDE := /home/wyl8899/lib/atlas/include
BLAS_LIB := /home/wyl8899/lib/atlas/lib
LIBRARIES += cblas atlas	
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

COMMON_FLAGS += -DCPU_ONLY
LIBRARIES += glog gflags protobuf leveldb snappy \
	lmdb boost_system hdf5_hl hdf5 m \
	opencv_core opencv_highgui opencv_imgproc
WARNINGS := -Wall -Wno-sign-compare

COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

all: $(EXEC)

$(EXEC): $(EXEC).o
	@ g++ $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(CAFFE_LIB) 
	
$(EXEC).o : $(EXEC).cpp
	@ g++ $< $(CXXFLAGS) -c -o $@
	
clean:
	rm $(EXEC).o $(EXEC)
