heatdist: heatdist.cu cudaFunctions.o
		nvcc -o heatdist -arch=sm_52 -rdc=true heatdist.cu cudaFunctions.o

cudaFunctions.o: cudaFunctions.cu
		nvcc -dc -Wno-deprecated-gpu-targets cudaFunctions.cu

.PHONY: clean
clean:
	rm heatdist cudaFunctions.o
