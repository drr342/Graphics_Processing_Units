#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <cuda.h>

void exit_with_help()
{
	printf(
	"Usage: svm-scale [options] data_filename\n"
	"options:\n"
	"-l lower : x scaling lower limit (default -1)\n"
	"-u upper : x scaling upper limit (default +1)\n"
	"-y y_lower y_upper : y scaling limits (default: no y scaling)\n"
	"-s save_filename : save scaling parameters to save_filename\n"
	"-r restore_filename : restore scaling parameters from restore_filename\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

#define BLOCK 1024
#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_node {
	int index;
	float value;
};

struct svm_problem {
	int l;
	float *y;
	struct svm_node **x;
};

char *line = NULL;
int max_line_len = 1024;
float lower=-1.0,upper=1.0,y_lower,y_upper;
int y_scaling = 0;
float *feature_max;
float *feature_min;
float y_min_max[2];
float y_max = -DBL_MAX;
float y_min = DBL_MAX;
int min_idx = INT_MAX;
int max_idx = INT_MIN;
long int num_nonzeros = 0;
long int new_num_nonzeros = 0;
FILE *fp, *fp_restore = NULL;
char *save_filename = NULL;
char *restore_filename = NULL;
char scale_filename[1024];
__device__ __managed__ struct svm_problem prob;		// set by read_problem
__device__ __managed__ struct svm_node * x_space;

char* readline(FILE *input);
void clean_up(FILE *fp_restore, FILE *fp, const char *msg);
void parse_command_line(int argc, char ** argv);
void read_problem(FILE * fp);
void save();
void restore();

__device__ float scale(float value, float min, float max, float lower, float upper);
__global__ void find_min (float * min_values, int min_idx, int max_idx);
__global__ void find_max (float * max_values, int min_idx, int max_idx);
__global__ void find_min_max_y (float * min_max_y);
__global__ void scale_problem (float * min_values, float * max_values, float * min_max_y,
								int min_idx, int max_idx, int iPerThread, int y_scaling,
								float lower, float upper, float y_lower, float y_upper);

int main(int argc,char **argv) {
	parse_command_line(argc, argv);
	read_problem(fp);

	cudaStream_t stream1, stream2, stream3;
	cudaError_t error;
	float *min_values, *max_values, *min_max_y;

	size_t size = (max_idx - min_idx + 1) * sizeof(float);
	feature_min = Malloc(float, size);
	feature_max = Malloc(float, size);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaMalloc(&min_values, size);
	cudaMalloc(&max_values, size);
	if (y_scaling) cudaMalloc(&min_max_y, 2 * sizeof(float));

	if (!restore_filename) {
		find_min<<<1,BLOCK,size,stream1>>>(min_values, min_idx, max_idx);
		cudaMemcpyAsync(feature_min, min_values, size, cudaMemcpyDeviceToHost, stream1);
		find_max<<<1,BLOCK,size,stream2>>>(max_values, min_idx, max_idx);
		cudaMemcpyAsync(feature_max, max_values, size, cudaMemcpyDeviceToHost, stream2);

		if (y_scaling) {
			cudaStreamCreate(&stream3);
			find_min_max_y<<<1,BLOCK,0,stream3>>>(min_max_y);
			cudaMemcpyAsync(y_min_max, min_max_y, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream3);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if(error != cudaSuccess) {
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			printf("Exiting program...\n");
			exit(-1);
		}
	} else {
		memset(feature_min, 0, size);
		memset(feature_max, 0, size);
		restore();
		cudaMemcpyAsync(min_values, feature_min, size, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(max_values, feature_max, size, cudaMemcpyHostToDevice, stream2);
		if (y_scaling) {
			cudaStreamCreate(&stream3);
			cudaMemcpyAsync(min_max_y, y_min_max, 2 * sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
	}

	if (y_scaling) {
		y_min = y_min_max[0];
		y_max = y_min_max[1];
	}

	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	int gridSize = (int)ceil((double)min(prop.maxGridSize[0], prob.l) / BLOCK);
	int iPerThread = (int)ceil((double)prob.l / (gridSize * BLOCK));
	scale_problem<<<gridSize,BLOCK,2*size,stream1>>>(min_values, max_values, min_max_y,
		min_idx, max_idx, iPerThread, y_scaling,
		lower, upper, y_lower, y_upper);
	if (save_filename)
		save();

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Exiting program...\n");
		exit(-1);
	}

	FILE * fp_scaled = fopen(scale_filename, "w");
	if (fp_scaled == NULL) {
		fprintf(stderr,"can't create output file %s\n", scale_filename);
		exit(1);
	}

	for (int i = 0; i < prob.l; i++) {
		fprintf(fp_scaled, "%.17g ", prob.y[i]);
		const svm_node * p = prob.x[i];
		while (p->index != -1) {
			if (p->value != 0) {
				fprintf(fp_scaled, "%d:%.6g ", p->index, p->value);
				new_num_nonzeros++;
			}
			p++;
		}
		fprintf(fp_scaled, "\n");
	}

	if (new_num_nonzeros > num_nonzeros)
	fprintf(stderr,
		"WARNING: original #nonzeros %ld\n"
		"       > new      #nonzeros %ld\n"
		"If feature values are non-negative and sparse, use -l 0 rather than the default -l -1\n",
		num_nonzeros, new_num_nonzeros);

	cudaFree(min_values);
	cudaFree(max_values);
	cudaStreamDestroy(stream2);
	if (y_scaling) {
		cudaFree(min_max_y);
		cudaStreamDestroy(stream3);
	}
	cudaFree(prob.y);
	cudaFree(prob.x);
	cudaFree(x_space);
	cudaStreamDestroy(stream1);
	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	fclose(fp_scaled);
	return 0;
}

__device__ static float atomicMin(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void find_min (float * min_values, int min_idx, int max_idx) {
	extern __shared__ float shared_min[];
	int i = threadIdx.x;

	// INITIALIZE SHARED MEMORY
	int size = max_idx - min_idx + 1;
	int items_per_thread = (int) ceil((float)size/BLOCK);
	int start = i * items_per_thread, end;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			shared_min[j] = FLT_MAX;
		}
	}
	__syncthreads();

	// POPULATE SHARED MEMORY
	size = prob.l;
	items_per_thread = (int) ceil((float)size/BLOCK);
	start = i * items_per_thread;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			const svm_node * p = prob.x[j];
			while (p->index != -1) {
				int index = p->index - min_idx;
				atomicMin(&shared_min[index], p->value);
				p++;
			}
		}
	}
	__syncthreads();

	// COPY TO GLOBAL MEMORY
	size = max_idx - min_idx + 1;
	items_per_thread = (int) ceil((float)size/BLOCK);
	start = i * items_per_thread;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			min_values[j] = shared_min[j];
		}
	}
}

__global__ void find_max (float * max_values, int min_idx, int max_idx) {
	extern __shared__ float shared_max[];
	int i = threadIdx.x;

	// INITIALIZE SHARED MEMORY
	int size = max_idx - min_idx + 1;
	int items_per_thread = (int) ceil((float)size/BLOCK);
	int start = i * items_per_thread, end;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			shared_max[j] = -FLT_MAX;
		}
	}
	__syncthreads();

	// POPULATE SHARED MEMORY
	size = prob.l;
	items_per_thread = (int) ceil((float)size/BLOCK);
	start = i * items_per_thread;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			const svm_node * p = prob.x[j];
			while (p->index != -1) {
				int index = p->index - min_idx;
				atomicMax(&shared_max[index], p->value);
				p++;
			}
		}
	}
	__syncthreads();

	// COPY TO GLOBAL MEMORY
	size = max_idx - min_idx + 1;
	items_per_thread = (int) ceil((float)size/BLOCK);
	start = i * items_per_thread;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			max_values[j] = shared_max[j];
		}
	}
}

__global__ void find_min_max_y(float * min_max_y) {
	__shared__ float shared_min, shared_max;
	int i = threadIdx.x;

	// INITIALIZE SHARED MEMORY
	if (i == 0) {
		shared_min = FLT_MAX;
		shared_max = -FLT_MAX;
	}
	__syncthreads();

	// POPULATE SHARED MEMORY
	int size = prob.l;
	int items_per_thread = (int) ceil((float)size/BLOCK);
	int start = i * items_per_thread, end;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			atomicMax(&shared_max, prob.y[j]);
			atomicMin(&shared_min, prob.y[j]);
		}
	}
	__syncthreads();

	// COPY TO GLOBAL MEMORY
	if (i == 0) {
		min_max_y[0] = shared_min;
		min_max_y[1] = shared_max;
	}
}

__global__ void scale_problem (float * min_values, float * max_values, float * min_max_y,
								int min_idx, int max_idx, int iPerThread, int y_scaling,
								float lower, float upper, float y_lower, float y_upper) {
	extern __shared__ float min_max[];

	// INITIALIZE SHARED MEMORY
	int i = threadIdx.x;
	int size = max_idx - min_idx + 1;
	int items_per_thread = (int)ceil((float)size/BLOCK);
	int start = i * items_per_thread, end;
	if (start < size) {
		end = min(start + items_per_thread, size);
		for (int j = start; j < end; j++) {
			min_max[j] = min_values[j];
			min_max[size + j] = max_values[j];
		}
	}
	__syncthreads();
	float * shared_min = min_max;
	float * shared_max = (float*)&min_max[size];

	// PERFORM SCALING
	for (int j = 0; j < iPerThread; j++) {
		i = blockIdx.x * blockDim.x + threadIdx.x + j * gridDim.x * blockDim.x;
		if (i >= prob.l) break;
		svm_node * p = prob.x[i];
		while (p->index != -1) {
			int idx = p->index - min_idx;
			p->value = scale(p->value, shared_min[idx], shared_max[idx], lower, upper);
			p++;
		}
		if (y_scaling) {
			prob.y[i] = scale(prob.y[i], min_max_y[0], min_max_y[1], y_lower, y_upper);
		}
	}
}

__device__ float scale(float value, float min, float max, float lower, float upper) {
	/* skip single-valued attribute */
	if(max == min || max == -FLT_MAX || min == FLT_MAX)
		return value;
	if(value == min)
		return lower;
	if(value == max)
		return upper;
	return lower + (upper-lower) *
			(value - min) / (max - min);
}

char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void clean_up(FILE *fp_restore, FILE *fp, const char* msg)
{
	fprintf(stderr,	"%s", msg);
	free(line);
	free(feature_max);
	free(feature_min);
	fclose(fp);
	fclose(fp_restore);
	exit(1);
}

void parse_command_line (int argc, char ** argv) {
	int i;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'l': lower = atof(argv[i]); break;
			case 'u': upper = atof(argv[i]); break;
			case 'y':
				y_lower = atof(argv[i]);
				++i;
				y_upper = atof(argv[i]);
				y_scaling = 1;
				break;
			case 's': save_filename = argv[i]; break;
			case 'r': restore_filename = argv[i]; break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	if(!(upper > lower) || (y_scaling && !(y_upper > y_lower)))
	{
		fprintf(stderr,"inconsistent lower/upper specification\n");
		exit(1);
	}

	if(restore_filename && save_filename)
	{
		fprintf(stderr,"cannot use -r and -s simultaneously\n");
		exit(1);
	}

	if(argc != i+1)
		exit_with_help();

	const char * suffix = ".scale.gpu";
	sprintf(scale_filename, "%s%s", argv[i], suffix);

	fp=fopen(argv[i],"r");

	if(fp==NULL)
	{
		fprintf(stderr,"can't open file %s\n", argv[i]);
		exit(1);
	}
}

void read_problem(FILE *fp) {
	int max_index, inst_max_index, i;
	size_t elements, j;
	char *endptr;
	char *idx, *val, *label;

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	cudaMallocManaged(&prob.y, prob.l * sizeof(float));
	cudaMallocManaged(&prob.x, prob.l * sizeof(struct svm_node *));
	cudaMallocManaged(&x_space, elements * sizeof(struct svm_node));

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if (x_space[j].index > max_idx)  max_idx = x_space[j].index;
			if (x_space[j].index < min_idx) min_idx = x_space[j].index;
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			num_nonzeros++;
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}
	rewind(fp);
}

void restore () {
	int idx, c;
	float fmin, fmax;

	fp_restore = fopen(restore_filename,"r");
	if(fp_restore==NULL)
	{
		fprintf(stderr,"can't open file %s\n", restore_filename);
		exit(1);
	}

	if((c = fgetc(fp_restore)) == 'y') {
		if(fscanf(fp_restore, "%f %f\n", &y_lower, &y_upper) != 2 ||
		   fscanf(fp_restore, "%f %f\n", &y_min, &y_max) != 2)
			clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
		y_scaling = 1;
	}
	else
		ungetc(c, fp_restore);

	if (fgetc(fp_restore) == 'x') {
		if(fscanf(fp_restore, "%f %f\n", &lower, &upper) != 2)
			clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
		while(fscanf(fp_restore,"%d %f %f\n",&idx,&fmin,&fmax) == 3) {
			int i = idx - min_idx;
			if (i < 0 || i > max_idx - min_idx) {
				continue;
			} else {
				feature_min[i] = fmin;
				feature_max[i] = fmax;
			}
		}
	} else {
		clean_up(fp_restore, fp, "ERROR: failed to read scaling parameters\n");
	}
	fclose(fp_restore);
}

void save() {
	FILE *fp_save = fopen(save_filename,"w");
	if(fp_save==NULL) {
		fprintf(stderr,"can't open file %s\n", save_filename);
		exit(1);
	}
	if(y_scaling) {
		fprintf(fp_save, "y\n");
		fprintf(fp_save, "%.17g %.17g\n", y_lower, y_upper);
		fprintf(fp_save, "%.17g %.17g\n", y_min, y_max);
	}
	fprintf(fp_save, "x\n");
	fprintf(fp_save, "%.17g %.17g\n", lower, upper);
	for (int i = 0; i < max_idx - min_idx + 1; i++) {
		if (feature_min[i] != feature_max[i] &&
		    feature_min[i] != FLT_MAX && feature_max[i] != -FLT_MAX)
			fprintf(fp_save,"%d %.17g %.17g\n",i+min_idx,feature_min[i],feature_max[i]);
	}
	fclose(fp_save);
}
