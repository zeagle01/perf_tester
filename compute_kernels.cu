
#include "compute_kernels.h"

#include "cuda_runtime.h"
#include "vector_add_cuda.h"

template<typename T>
__global__ void vector_add(int num, T* c, T* a, T* b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num)
	{
		c[tid] = a[tid] + b[tid];
	}
}


template<typename T>
void  CUDA_Add<T>::apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row)
{

	T* in0 = &in[0 * out_col];
	T* in1 = &in[1 * out_col];

	int tpb = 128;
	int bpg = (out_col - 1) / tpb + 1;
	vector_add << <bpg, tpb >> > (out_col, out, in0, in1);
}


template CUDA_Add<float>;
template CUDA_Add<double>;
