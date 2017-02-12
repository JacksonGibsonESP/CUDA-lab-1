#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h> 
//#include <chrono>

using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define BLOCK_SIZE 1024

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
		    }															\
} while (0)

__global__ void kernel_main(double * matrix, unsigned int i, unsigned int height, unsigned int width, unsigned int * index, int *index_of_max, unsigned int offset)
{
	double max = 0.0000001;
	*index_of_max = -1;
	for (unsigned int j = offset; j < height; j++)
	{
		if (fabs(matrix[index[j] * width + i]) > max)
		{
			max = fabs(matrix[index[j] * width + i]);
			*index_of_max = j;
		}
	}

	if (*index_of_max != -1)
	{
		unsigned int tmp = index[*index_of_max];
		index[*index_of_max] = index[offset];
		index[offset] = tmp;
	}
}

//__device__ double ratio_ = 0;

/*__global__ void kernel_count_ratio(double * matrix, unsigned int i, unsigned int l, unsigned int height, unsigned int width, unsigned int * index, unsigned int offset)
{
	ratio_ = matrix[index[l] * width + i] / matrix[index[(offset - 1)] * width + i];
}*/

__global__ void kernel_count_ratios(double * matrix, unsigned int i, unsigned int height, unsigned int width, unsigned int * index, unsigned int offset, double * ratios)
{
/*	double tmp = matrix[index[(offset - 1)] * width + i];
	for (unsigned int l = offset; l < height; l++)
	{
		ratios[l] = matrix[index[l] * width + i] / tmp;
	}
*/
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ double tmp;
	tmp = matrix[index[(offset - 1)] * width + i];
	while (offset + tid < height)
	{
		ratios[offset + tid] = matrix[index[offset + tid] * width + i] / tmp;
		tid += blockDim.x * gridDim.x;
	}
}

/*__global__ void kernel_rows_substraction(double * matrix, unsigned int i, unsigned int l, unsigned int width, unsigned int * index, unsigned int offset)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (i + tid < width)
	{
		matrix[index[l] * width + i + tid] -= ratio_ * matrix[index[(offset - 1)] * width + i + tid];
		tid += blockDim.x * gridDim.x;
	}
}*/

__global__ void kernel_rows_substraction(double * matrix, unsigned int i, unsigned int height, unsigned int width, unsigned int * index, unsigned int offset, double * ratios)
{
	/*for (unsigned int l = offset; l < height; l++)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		double ratio_ = ratios[l];
		while (i + tid < width)
		{
			matrix[index[l] * width + i + tid] -= ratio_ * matrix[index[(offset - 1)] * width + i + tid];
			tid += blockDim.x * gridDim.x;
		}
	}*/
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (i + tid < width)
	{
		double factor = matrix[index[(offset - 1)] * width + i + tid];
		for (unsigned int l = offset; l < height; l++)
		{
			matrix[index[l] * width + i + tid] -= ratios[l] * factor;
		}
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_rank_count(double * matrix, unsigned int height, unsigned int width, unsigned int * index, unsigned int *rank)
{
	unsigned int i = 0, j = 0;
	//*rank = 0;
	unsigned int rank_ = 0;
	while (true)
	{
		if (fabs(matrix[index[i] * width + j]) > 0.0000001)
		{
			rank_++;
			i++;
			j++;
			if (i >= height || j >= width)
				break;
		}
		else
		{
			j++;
			if (j >= width)
				break;
		}
	}
	(*rank) = rank_;
}

int main()
{
	unsigned int height, width;
	scanf("%ud", &height);
	scanf("%ud", &width);

	if (height == 0 || width == 0)
	{
		cout << "ERROR: incorrect data\n";
		return 0;
	}
	if (height == 1 && width == 1)
	{
		double tmp;
		scanf("%lf", &tmp);
		if (fabs(tmp) > 0.0000001)
		{
			cout << 1 << '\n';
		}
		else
		{
			cout << 0 << '\n';
		}
		return 0;
	}

	//double * matrix = new double[height * width];
	double * matrix = (double *)malloc(sizeof(double) * height * width);

	for (unsigned int i = 0; i < height; i++)
	{
		for (unsigned int j = 0; j < width; j++)
		{
			double tmp;
			//cin >> tmp;
			scanf("%lf", &tmp);
			matrix[i * width + j] = tmp;
		}
	}
	
	double * dev_matrix;
	CSC(cudaMalloc(&dev_matrix, sizeof(double) * height * width));
	CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * height * width, cudaMemcpyHostToDevice));

	//unsigned int * index = new unsigned int[height];
	unsigned int * index = (unsigned int *)malloc(sizeof(unsigned int) * height);
	for (unsigned int i = 0; i < height; i++)
	{
		index[i] = i;
	}

	unsigned int * dev_index;
	CSC(cudaMalloc(&dev_index, sizeof(unsigned int) * height));
	CSC(cudaMemcpy(dev_index, index, sizeof(unsigned int) * height, cudaMemcpyHostToDevice));

	int host_index_of_max;
	int * device_index_of_max;
	CSC(cudaMalloc(&device_index_of_max, sizeof(int)));

	unsigned int threads_count = BLOCK_SIZE;
	unsigned int blocks_count = MAX(width, height) / threads_count + 1;

	unsigned int offset = 0;
	/*cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start, 0));*/

	double * dev_ratios;
	CSC(cudaMalloc(&dev_ratios, sizeof(double) * height));
	//auto start_time = chrono::high_resolution_clock::now();
	for (unsigned int i = 0; i < width; i++)
	{
		kernel_main << < 1, 1 >> > (dev_matrix, i, height, width, dev_index, device_index_of_max, offset);
		CSC(cudaMemcpy(&host_index_of_max, device_index_of_max, sizeof(int), cudaMemcpyDeviceToHost));
		if (host_index_of_max != -1)
		{
			offset++;
			kernel_count_ratios << < height / threads_count + 1, threads_count >> >(dev_matrix, i, height, width, dev_index, offset, dev_ratios);
			kernel_rows_substraction << < blocks_count, threads_count >> > (dev_matrix, i, height, width, dev_index, offset, dev_ratios);
		}
	}
	
	unsigned int * dev_rank;
	CSC(cudaMalloc(&dev_rank, sizeof(unsigned int)));
	kernel_rank_count << < 1, 1 >> > (dev_matrix, height, width, dev_index, dev_rank);
	/*CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));*/

	unsigned int rank;
	CSC(cudaMemcpy(&rank, dev_rank, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cout << rank << '\n';

/*	CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * height * width, cudaMemcpyDeviceToHost));

	CSC(cudaMemcpy(index, dev_index, sizeof(unsigned int) * height, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < height; i++)
	{
		for (unsigned int j = 0; j < width; j++)
		{
			cout << matrix[index[i] * width + j] << ' ';
		}
		cout << '\n';
	}*/
	//delete matrix;
	//delete index;
	free(matrix);
	free(index);
	CSC(cudaFree(dev_matrix));
	CSC(cudaFree(dev_index));
	CSC(cudaFree(dev_rank));
	//auto end_time = chrono::high_resolution_clock::now();
	//cout << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "\n";
	cudaProfilerStop();
	return 0;
}
