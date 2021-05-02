
#pragma once

#include "cuda_runtime.h"
#include "compute_kernels.h"
#include "compute_kernels.h"


template<typename T, typename Kernel> __global__ void cuda_for_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		Kernel::apply(in, out, in_col, in_row, out_col, out_row, tid);
	}
}


template< typename T, typename Kernel, int thread_per_block>
void cuda_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
{
	int tpb = thread_per_block;
	int bpg = (size - 1) / tpb + 1;
	cuda_for_loop<T, Kernel> << <bpg, tpb >> > (in, out, size, in_col, in_row, out_col, out_row);
}

#define DELEARE_CUDA_LOOP_TEMPLATE(type,kernel,tpb)\
template void cuda_loop<type, kernel<type>,tpb>(type* in, type* out, int size, int in_col, int in_row, int out_col, int out_row);\

DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 16)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 32)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 64)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 128)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 256)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 512)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, 1024)

DELEARE_CUDA_LOOP_TEMPLATE(float, Mupltiply_Add_N_Times_Kernel, 128)
DELEARE_CUDA_LOOP_TEMPLATE(double, Mupltiply_Add_N_Times_Kernel, 128)


