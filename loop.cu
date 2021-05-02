
#pragma once

#include "cuda_runtime.h"
#include "compute_kernels.h"
#include "problems.h"
#include "devices.h"


template<typename T, typename Kernel> __global__ void cuda_for_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		Kernel::apply(in, out, in_col, in_row, out_col, out_row, tid);
	}
}


template< typename T, typename Kernel, typename Launch_Config>
void cuda_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
{
	int tpb = Launch_Config::thread_per_block;
	int bpg = (size - 1) / tpb + 1;
	cuda_for_loop<T, Kernel> << <bpg, tpb >> > (in, out, size, in_col, in_row, out_col, out_row);
}

#define DELEARE_CUDA_LOOP_TEMPLATE(type,kernel,lauch_config,...)\
template void cuda_loop<type, kernel<type,__VA_ARGS__>,lauch_config>(type* in, type* out, int size, int in_col, int in_row, int out_col, int out_row);\

DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<16>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<32>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<64>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<128>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<256>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<512>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Add_Kernel, Launch_Config<1024>)

DELEARE_CUDA_LOOP_TEMPLATE(float, Mupltiply_Add_N_Times_Kernel, Launch_Config<128>, Repeat<10>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Mupltiply_Add_N_Times_Kernel, Launch_Config<128>, Repeat<100>)
DELEARE_CUDA_LOOP_TEMPLATE(float, Mupltiply_Add_N_Times_Kernel, Launch_Config<128>, Repeat<1000>)
DELEARE_CUDA_LOOP_TEMPLATE(double, Mupltiply_Add_N_Times_Kernel, Launch_Config<128>, Repeat<100>)
DELEARE_CUDA_LOOP_TEMPLATE(double, Mupltiply_Add_N_Times_Kernel, Launch_Config<128>, Repeat<1000>)


