#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <cuda.h>
#include <locale.h>

#include "svm.h"

int print_null(const char *s,...) {return 0;}
static int (*info)(const char *fmt,...) = &printf;

void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

#define BLOCK 32
#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char *line = NULL;
static int max_line_len;
int max_nr_attr = 64;
int predict_probability = 0;
FILE *input, *output;
__device__ __managed__ struct svm_model* model;
__device__ __managed__ struct svm_problem prob;		// set by read_problem
__device__ __managed__ struct svm_node * x_space;

void parse_command_line (int argc, char ** argv);
void read_problem();
void predict ();
static char* readline (FILE *input);

__global__ void gpu_predict (struct svm_model * model, struct svm_node ** x, struct memory * mem,
								double * predict_label, int iPerThread);
__global__ void gpu_predict_probability (struct svm_model * model, struct svm_node ** x, struct memory * mem,
											double ** prob_estimates, double * predict_label, int iPerThread);

int main(int argc, char **argv) {
	parse_command_line(argc, argv);
	read_problem();
	predict();
	cuda_destroy_model(model);
	cudaFree(model);
	free(prob.y);
	cudaFree(prob.x);
	cudaFree(x_space);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}

void parse_command_line (int argc, char ** argv) {
	// parse options
	int i;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i>=argc-1)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	const char * preffix = (i+2 < argc) ? argv[i+2] : argv[i];
	char output_filename[1024];
	sprintf(output_filename,"%s.predict.gpu", preffix);
	output = fopen(output_filename,"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",output_filename);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1],1))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	if(predict_probability)
	{
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}
}

void read_problem() {
	int max_index, inst_max_index, i;
	size_t elements, j;
	char *endptr;
	char *idx, *val, *label;

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(input)!=NULL)
	{
		char *p = strtok(line," \t"); // labe
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
	rewind(input);

	prob.y = Malloc(double, prob.l);
	cudaMallocManaged(&prob.x, prob.l * sizeof(struct svm_node *));
	cudaMallocManaged(&x_space, elements * sizeof(struct svm_node));

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(input);
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
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}
}

void predict()
{
	int correct = 0;
	int total = prob.l;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type = model->param.svm_type;
	int nr_class = model->nr_class;
	double **prob_estimates=NULL;
	double *predict_label;
	struct memory * mem = NULL;

	cudaMallocManaged(&predict_label, prob.l * sizeof(double));

	if (predict_probability) {
		if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else {
			cudaMallocManaged(&prob_estimates, prob.l * sizeof(double*));
			for (int i = 0; i < prob.l; i++)
				cudaMallocManaged(&prob_estimates[i], nr_class * sizeof(double));
			fprintf(output,"labels");
			for (int j = 0; j < nr_class; j++)
				fprintf(output," %d", model->label[j]);
			fprintf(output,"\n");
		}
	}

	if(!(svm_type == ONE_CLASS || svm_type == EPSILON_SVR || svm_type == NU_SVR)) {
		cudaMallocManaged(&mem, prob.l*sizeof(struct memory));
		for (int i = 0; i < prob.l; i++) {
			cudaMallocManaged(&mem[i].dec_values,sizeof(double)*nr_class*(nr_class-1)/2);
			cudaMallocManaged(&mem[i].kvalue,sizeof(double)*model->l);
			cudaMallocManaged(&mem[i].start,sizeof(int)*nr_class);
			cudaMallocManaged(&mem[i].vote,sizeof(int)*nr_class);
			mem[i].pairwise_prob=NULL;
			mem[i].Q=NULL;
			mem[i].Qp=NULL;
		}
		if (predict_probability &&
				(svm_type==C_SVC || svm_type==NU_SVC) &&
				model->probA != NULL && model->probB != NULL) {
			for (int i = 0; i < prob.l; i++) {
				cudaMallocManaged(&mem[i].pairwise_prob,sizeof(double*)*nr_class);
				if (nr_class != 2) {
					cudaMallocManaged(&mem[i].Q,sizeof(double*)*nr_class);
					cudaMallocManaged(&mem[i].Qp,sizeof(double)*nr_class);
				}
				for (int j = 0; j < nr_class; j++) {
					cudaMallocManaged(&mem[i].pairwise_prob[j],sizeof(double)*nr_class);
					if (nr_class != 2)
						cudaMallocManaged(&mem[i].Q[j],sizeof(double)*nr_class);
				}
			}
		}
	}

	cudaDeviceSynchronize();
	cudaError_t cuda_rror = cudaGetLastError();
	if(cuda_rror != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(cuda_rror));
		printf("Exiting program...\n");
		exit(-1);
	} /*else {
		printf("Allocation completed successfully\n");
	}*/

	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	int gridSize = (int)ceil((double)min(prop.maxGridSize[0], prob.l) / BLOCK);
	int iPerThread = (int)ceil((double)prob.l / (gridSize * BLOCK));
	if (predict_probability &&
		(svm_type==C_SVC || svm_type==NU_SVC) &&
		model->probA != NULL && model->probB != NULL)
		gpu_predict_probability<<<gridSize, BLOCK>>>(model, prob.x, mem, prob_estimates, predict_label, iPerThread);
	else
		gpu_predict<<<gridSize, BLOCK>>>(model, prob.x, mem, predict_label, iPerThread);
	cudaDeviceSynchronize();
	cuda_rror = cudaGetLastError();
	if(cuda_rror != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(cuda_rror));
		printf("Exiting program...\n");
		exit(-1);
	} /*else {
		printf("Prediction completed successfully\n");
	}*/

	for (int i = 0; i < total; i++) {
		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC)) {
			fprintf(output,"%g",predict_label[i]);
			for(int j = 0; j < nr_class; j++)
				fprintf(output," %g",prob_estimates[i][j]);
			fprintf(output,"\n");
		}
		else {
			fprintf(output,"%.17g\n",predict_label[i]);
		}
		if(predict_label[i] == prob.y[i])
			correct++;
		error += (predict_label[i]-prob.y[i])*(predict_label[i]-prob.y[i]);
		sump += predict_label[i];
		sumt += prob.y[i];
		sumpp += predict_label[i]*predict_label[i];
		sumtt += prob.y[i]*prob.y[i];
		sumpt += predict_label[i]*prob.y[i];
	}

	if (svm_type==NU_SVR || svm_type==EPSILON_SVR) {
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);

	cudaFree(predict_label);
	if(predict_probability) {
		for (int i = 0; i < prob.l; i++)
			cudaFree(prob_estimates[i]);
		cudaFree(prob_estimates);
	}
	if(!(svm_type == ONE_CLASS || svm_type == EPSILON_SVR || svm_type == NU_SVR)) {
		if (predict_probability &&
				(svm_type==C_SVC || svm_type==NU_SVC) &&
				model->probA != NULL && model->probB != NULL) {
			for (int i = 0; i < prob.l; i++) {
				for (int j = 0; j < nr_class; j++) {
					cudaFree(mem[i].pairwise_prob[j]);
					if (nr_class != 2)
						cudaFree(mem[i].Q[j]);
				}
				cudaFree(mem[i].pairwise_prob);
				if (nr_class != 2) {
					cudaFree(mem[i].Q);
					cudaFree(mem[i].Qp);
				}
			}
		}
		for (int i = 0; i < prob.l; i++) {
			cudaFree(mem[i].dec_values);
			cudaFree(mem[i].kvalue);
			cudaFree(mem[i].start);
			cudaFree(mem[i].vote);
		}
		cudaFree(mem);
	}
}

__global__ void gpu_predict (struct svm_model * model, struct svm_node ** x, struct memory * mem,
								double * predict_label, int iPerThread) {
	for (int j = 0; j < iPerThread; j++) {
		int i = blockIdx.x * blockDim.x + threadIdx.x + j * gridDim.x * blockDim.x;
		if (i >= prob.l) break;
		struct memory * thread_mem = (mem) ? &mem[i] : NULL;
		predict_label[i] = svm_predict(model, x[i], thread_mem);
	}
}

__global__ void gpu_predict_probability (struct svm_model * model, struct svm_node ** x, struct memory * mem,
											double ** prob_estimates, double * predict_label, int iPerThread) {
	for (int j = 0; j < iPerThread; j++) {
		int i = blockIdx.x * blockDim.x + threadIdx.x + j * gridDim.x * blockDim.x;
		if (i >= prob.l) break;
		predict_label[i] = svm_predict_probability(model, x[i], prob_estimates[i], &mem[i]);
	}
}

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
