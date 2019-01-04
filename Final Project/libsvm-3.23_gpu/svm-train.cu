#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <cuda.h>

#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define SUB_PROB 32
#define GRID_SIZE 1

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

int classes_index (double * d, int total, double y);
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation ();

void get_classes_count (double ** classes, int ** count, int * total);
void allocate_final_model ();

__global__ void gpu_train (int * fm_k, int * fm_l, int * fm_space, int * fm_mask);
__global__ void create_subProblems (double * classes, int * count, int * prob_flags, int total);
__global__ void delete_subProblems ();
__global__ void copy_final_model ();

__device__ struct svm_parameter d_param;
__device__ struct svm_problem subproblems[SUB_PROB];
__device__ struct svm_model * models[SUB_PROB];
__device__ __managed__ struct svm_model final_model;
__device__ __managed__ int fm_k, fm_l, fm_space, fm_mask;
__device__ __managed__ struct svm_problem prob;		// set by read_problem
__device__ __managed__ struct svm_node * x_space;
struct svm_parameter param;	// set by parse_command_line
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

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

__global__ void pass_param (struct svm_parameter * temp_param) {
	d_param = *temp_param;
}

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);

	struct svm_parameter * temp_param;
	cudaMalloc(&temp_param, sizeof(struct svm_parameter));
	cudaMemcpy(temp_param, &param, sizeof(struct svm_parameter), cudaMemcpyHostToDevice);
	pass_param<<<1,1>>>(temp_param);
	cudaDeviceSynchronize();
	cudaFree(temp_param);

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Exiting program...\n");
		exit(-1);
	} else {
		debug("After pass params", DEBUG, 0);
	}


	double * classes;
	int * count, total;
	get_classes_count(&classes, &count, &total);

	double * d_classes;
	int * d_count, * prob_flags;
	cudaMalloc(&d_classes, total * sizeof(double));
	cudaMalloc(&d_count, total * sizeof(int));
	cudaMallocManaged(&prob_flags, prob.l * sizeof(int));
	cudaMemcpy(d_classes, classes, total * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_count, count, total * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(prob_flags, 0, prob.l * sizeof(int));
	create_subProblems<<<GRID_SIZE, SUB_PROB>>>(d_classes, d_count, prob_flags, total);
	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		printf("Exiting program...\n");
		exit(-1);
	} else {
		debug("After create sub problems", DEBUG, 0);
	}

	cudaFree(d_classes);
	cudaFree(d_count);
	cudaFree(prob_flags);
	free(classes);
	free(count);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	if(cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		printf("Training models...\n");
		gpu_train<<<GRID_SIZE, SUB_PROB>>>(&fm_k, &fm_l, &fm_space, &fm_mask);
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if(error != cudaSuccess) {
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			printf("Exiting program...\n");
			exit(-1);
		} else {
			debug("After Train!!", DEBUG, 0);
		}

		int max = fm_k * (fm_k - 1) / 2;
		if (fm_k > max) max = fm_k;
		if (fm_l > max) max = fm_l;
		dim3 dimGrid((int)ceil(max / 1024.0));
		dim3 dimBlock(1024);
		allocate_final_model();
		// printf("Copying final model to host...\n");
		copy_final_model<<<dimGrid,dimBlock>>>();
		cudaDeviceSynchronize();

		if(svm_save_model(model_file_name, &final_model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}

		cuda_destroy_model(&final_model);
		error = cudaGetLastError();
		if(error != cudaSuccess) {
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			printf("Exiting program...\n");
			exit(-1);
		} else {
			debug("After save model", DEBUG, 0);
		}

	}

	delete_subProblems<<<GRID_SIZE, SUB_PROB>>>();
	cudaDeviceSynchronize();

	svm_destroy_param(&param);
	cudaFree(prob.y);
	cudaFree(prob.x);
	cudaFree(x_space);
	free(line);

	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		sprintf(model_file_name,"%s.gpu",argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model.gpu",p);
	}
}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

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

	cudaMallocManaged(&prob.y, prob.l * sizeof(double));
	cudaMallocManaged(&prob.x, prob.l * sizeof(struct svm_node*));
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

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

int classes_index (double * d, int total, double y) {
	for (int i = 0; i < total; i++) {
		if (d[i] == y) return i;
	}
	return -1;
}

// number of classes (total) and count of examples of each class (count)
void get_classes_count (double ** classes_ret, int ** count_ret, int * total_ret) {
	int size = 16;
	double * classes = Malloc(double, size);
	int * count = (int *) calloc(size, sizeof(int));
	int total = 0;
	for (int j = 0; j < prob.l; j++) {
		int i = classes_index(classes, total, prob.y[j]);
		if (i != -1) {
			count[i]++;
		} else {
			classes[total] = prob.y[j];
			count[total]++;
			total++;
			if (total == size) {
				size *= 2;
				classes = (double *) realloc(classes, size * sizeof(double));
				count = (int *) realloc(count, size * sizeof(int));
			}
		}
	}
	*classes_ret = classes;
	*count_ret = count;
	*total_ret = total;
}

__global__ void create_subProblems (double * classes, int * count, int * prob_flags, int total) {
	int i =  blockIdx.x * blockDim.x + threadIdx.x;

	// number of examples per class
	int * count_per_problem = Malloc(int, total);
	for (int j = 0; j < total; j++) {
		int r = count[j] % SUB_PROB;
		count_per_problem[j] = count[j] / SUB_PROB;
		if (i < r) count_per_problem[j]++;
	}

	// allocate subproblem
	int sum = 0;
	for (int j = 0; j < total; j++) {
		sum += count_per_problem[j];
	}
	subproblems[i].l = sum;
	subproblems[i].y = Malloc(double, sum);
	subproblems[i].x = Malloc(struct svm_node*, sum);

	// populate subproblem
	int start = i, counter = 0;
	while (counter != sum) {
		if (prob_flags[start] != 0) {
			start = (start + 1) % prob.l;
			continue;
		}
		atomicAdd(&prob_flags[start], 1);
		int class_index = -1;
		for (int j = 0; j < total; j++) {
			if (classes[j] == prob.y[start]) {
				class_index = j;
				break;
			}
		}
		if (count_per_problem[class_index] != 0) {
			count_per_problem[class_index]--;
			subproblems[i].y[counter] = prob.y[start];
			subproblems[i].x[counter] = prob.x[start];
			counter++;
		} else {
			atomicSub(&prob_flags[start], 1);
		}
		start = (start + 1) % prob.l;
	}
	free(count_per_problem);

	// allocate subproblem
	// int l = prob.l / SUB_PROB;
	// subproblems[i].l = l;
	// subproblems[i].y = Malloc(double, l);
	// subproblems[i].x = Malloc(struct svm_node*, l);

	// populate subproblems
	// for (int j = i * l, k = 0; k < l; j++, k++) {
	// 	subproblems[i].y[k] = prob.y[j];
	// 	subproblems[i].x[k] = prob.x[j];
	// }

}

__global__ void delete_subProblems () {
	int i =  blockIdx.x * blockDim.x + threadIdx.x;
	if (subproblems[i].y) {
		free(subproblems[i].y);
		free(subproblems[i].x);
	}
}

__global__ void gpu_train (int * fm_k, int * fm_l, int * fm_space, int * fm_mask) {
	int i =  blockIdx.x * blockDim.x + threadIdx.x;
	for (int k = 1; k <= SUB_PROB; k*=2) {
		if (i < SUB_PROB/k) {
			models[i] = svm_train(&subproblems[i], &d_param);
		}
		__syncthreads();
		if (i < SUB_PROB/(2*k)) {
			int offset = SUB_PROB/(2*k);
			// if (models[i]->l + models[i+offset]->l != 0) {
			subproblems[i].l = models[i]->l + models[i+offset]->l;
			free(subproblems[i].y);
			free(subproblems[i].x);
			subproblems[i].y = Malloc(double, subproblems[i].l);
			subproblems[i].x = Malloc(struct svm_node*, subproblems[i].l);
			int index1 = 0;
			for (int cl = 0; cl < models[i]->nr_class; cl++) {
				for (int j = 0; j < models[i]->nSV[cl]; j++) {
					subproblems[i].y[index1] = models[i]->label[cl];
					subproblems[i].x[index1] = models[i]->SV[index1];
					index1++;
				}
			}
			int index2 = 0;
			for (int cl = 0; cl < models[i+offset]->nr_class; cl++) {
				for (int j = 0; j < models[i+offset]->nSV[cl]; j++) {
					subproblems[i].y[index1] = models[i+offset]->label[cl];
					subproblems[i].x[index1] = models[i+offset]->SV[index2];
					index1++;
					index2++;
				}
			}
			// }
		}
		__syncthreads();
		int start = (k != SUB_PROB) ? 0 : 1;
		if (i >= start && i < SUB_PROB/k) {
			svm_free_and_destroy_model(&models[i]);
		}
		__syncthreads();
	}
	if (i == 0) {
		int space, max = 0;
		const svm_node * p = models[0]->SV[0];
		for (int j = 0; j < models[0]->l; j++) {
			space = 0;
			while (p->index != -1) {
				space++;
				p++;
			}
			if (space > max) max = space;
		}
		int mask = 0;
		if (models[0]->label) mask += 1;
		if (models[0]->probA) mask += 2;
		if (models[0]->probB) mask += 4;
		*fm_k = models[0]->nr_class;
		*fm_l = models[0]->l;
		*fm_space = max + 1;
		*fm_mask = mask;
	}
}

void allocate_final_model () {
	int fm_t = fm_k * (fm_k - 1) / 2;
	struct svm_node ** SV;
	double ** sv_coef, * rho, * probA = NULL, * probB = NULL;
	int * sv_indices, * label = NULL, * nSV;

	cudaMallocManaged(&SV, fm_l * sizeof(struct svm_node*));
	for (int i = 0; i < fm_l; i++)
		cudaMallocManaged(&SV[i], fm_space * sizeof(struct svm_node));
	cudaMallocManaged(&sv_coef, (fm_k-1) * sizeof(double*));
	for (int i = 0; i < fm_k - 1; i++)
		cudaMallocManaged(&sv_coef[i], fm_l * sizeof(double));
	cudaMallocManaged(&rho, fm_t * sizeof(double));
	if (fm_mask & 2)
		cudaMallocManaged(&probA, fm_t * sizeof(double));
	if (fm_mask & 4)
		cudaMallocManaged(&probB, fm_t * sizeof(double));
	cudaMallocManaged(&sv_indices, fm_l * sizeof(int));
	if (fm_mask & 1)
		cudaMallocManaged(&label, fm_k * sizeof(int));
	cudaMallocManaged(&nSV, fm_k * sizeof(int));

	final_model.SV = SV;
	final_model.sv_coef = sv_coef;
	final_model.rho = rho;
	final_model.probA = probA;
	final_model.probB = probB;
	final_model.sv_indices = sv_indices;
	final_model.label = label;
	final_model.nSV = nSV;

	final_model.param.svm_type = param.svm_type;
	final_model.param.kernel_type = param.kernel_type;
	final_model.param.degree = param.degree;
	final_model.param.gamma = param.gamma;
	final_model.param.coef0 = param.coef0;
	final_model.nr_class = fm_k;
	final_model.l = fm_l;
}

__global__ void copy_final_model() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < fm_l) {
		for(int j = 0; j< fm_k - 1; j++)
			final_model.sv_coef[j][i] = models[0]->sv_coef[j][i];
		const svm_node *src = models[0]->SV[i];
		svm_node *dst = final_model.SV[i];
		while(src->index != -1) {
			dst->index = src->index;
			dst->value = src->value;
			src++;
			dst++;
		}
		dst->index = -1;
		final_model.sv_indices[i] = models[0]->sv_indices[i];
	}
	if (i < fm_k) {
		if (models[0]->label)
			final_model.label[i] = models[0]->label[i];
		if (models[0]->nSV)
			final_model.nSV[i] = models[0]->nSV[i];
	}
	if (i < fm_k*(fm_k-1)/2) {
		final_model.rho[i] = models[0]->rho[i];
		if (models[0]->probA)
			final_model.probA[i] = models[0]->probA[i];
		if (models[0]->probB)
			final_model.probB[i] = models[0]->probB[i];
	}
	if (i == 0) {
		svm_free_and_destroy_model(&models[0]);
		final_model.free_sv = 1;
	}
}
