PROJECT := caffe

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

ALE_ROOT := /home/wyl8899/projects/ale
ALE_INCLUDE := $(ALE_ROOT)/src
ALE_LIB := $(ALE_ROOT)
INCLUDE_DIRS += $(ALE_INCLUDE)
LIBRARY_DIRS += $(ALE_LIB)
COMMON_FLAGS += -lale

COMMON_FLAGS += -DCPU_ONLY
LIBRARIES += glog gflags protobuf leveldb snappy \
	lmdb boost_system hdf5_hl hdf5 m \
	opencv_core opencv_highgui opencv_imgproc
WARNINGS := -Wall -Wno-sign-compare

SDL_INCLUDE := /usr/include/SDL
INCLUDE_DIRS += $(SDL_INCLUDE)
CXXFLAGS += -D__USE_SDL
COMMON_FLAGS += -lSDL

COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

# Uncomment to enable quiet compilation
Q = @

EXEC := dqn
SRC_DIR := ./src
SRC_OBJECTS := $(patsubst %.cpp, %.o, $(wildcard $(SRC_DIR)/*.cpp))
INCLUDE_DIR := ./include
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)

all: $(EXEC)

$(EXEC): $(SRC_OBJECTS)
	$(Q) g++ $< -o $@ $(LINKFLAGS) -l$(PROJECT) $(LDFLAGS) \
		-Wl,-rpath,$(CAFFE_LIB) 
	
%.o : %.cpp $(HEADERS)
	$(Q) g++ $< $(CXXFLAGS) -c -o $@
	
test: all
	./dqn --solver=n_bandit/solver.prototxt

clean:
	rm $(EXEC)
	rm $(SRC_OBJECTS)

