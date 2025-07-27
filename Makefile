# Compiler
CXX = nvcc

# Flags
CXXFLAGS = -std=c++17 -O2 -Wno-deprecated-gpu-targets

# OpenCV
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS := $(shell pkg-config --libs opencv4)

# CUDA
CUDA_LIBS = -lcuda -lcudnn -lcutensor -lcublas
CUDA_INC = -I/usr/local/cuda/include

# Source files
UTILS_SRC = utils.cu
UTILS_OBJ = utils.o

# Targets
TARGETS = train.exe test.exe

all: clean build

build: $(TARGETS)

train.exe: train.o $(UTILS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LDFLAGS) $(CUDA_LIBS)

test.exe: test.o $(UTILS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LDFLAGS) $(CUDA_LIBS)

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(CUDA_INC) -c $< -o $@

run-train:
	./train.exe $(ARGS)

run-test:
	./test.exe $(ARGS)

clean:
	rm -f *.o *.exe output*.txt
