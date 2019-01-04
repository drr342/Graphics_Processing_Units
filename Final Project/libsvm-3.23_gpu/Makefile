CXX ?= g++
CFLAGS = -Wall -O3 -fPIC

NVCC = nvcc
NVCCFLAGS = -arch=sm_70 -Wno-deprecated-gpu-targets -ccbin=$(CXX) --compiler-options='$(CFLAGS)'
# DEBUGFLAGS = -lineinfo
DEBUGFLAGS =
COMPATFLAGS = -gencode=arch=compute_60,code=sm_60 \
				 -gencode=arch=compute_61,code=sm_61 \
				 -gencode=arch=compute_70,code=sm_70 \
				 -gencode=arch=compute_70,code=compute_70

all: svm-train svm-predict svm-scale svm-train-gpu svm-predict-gpu svm-scale-gpu

svm-predict: svm-predict.cpp svm.o
	$(NVCC) $(NVCCFLAGS) svm-predict.cpp svm.o -o svm-predict
svm-train: svm-train.cpp svm.o
	$(NVCC) $(NVCCFLAGS) svm-train.cpp svm.o -o svm-train
svm-scale: svm-scale.c
	$(NVCC) $(NVCCFLAGS) svm-scale.c -o svm-scale
svm-predict-gpu: svm-predict.cu svm.o
	$(NVCC) $(NVCCFLAGS) $(DEBUGFLAGS) $(COMPATFLAGS) -rdc=true -o svm-predict-gpu svm-predict.cu svm.o
svm-train-gpu: svm-train.cu svm.o
	$(NVCC) $(NVCCFLAGS) $(DEBUGFLAGS) $(COMPATFLAGS) -rdc=true -o svm-train-gpu svm-train.cu svm.o
svm-scale-gpu: svm-scale.cu
	$(NVCC) $(NVCCFLAGS) $(DEBUGFLAGS) $(COMPATFLAGS) -o svm-scale-gpu svm-scale.cu
svm.o: svm.cu
	$(NVCC) $(NVCCFLAGS) $(DEBUGFLAGS) $(COMPATFLAGS) -dc svm.cu

.PHONY: clean
clean:
	rm -f *~ svm-scale-gpu svm-train-gpu svm-predict-gpu svm-train svm-predict svm-scale svm.o
