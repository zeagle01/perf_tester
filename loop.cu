
#pragma once

#include "cuda_runtime.h"
#include "compute_kernels.h"
#include "problems.h"
#include "devices.h"
#include "launch_config.h"
#include <iostream>
#include <vector>


template<typename T, typename Kernel > __global__ void cuda_for_loop(typename Kernel::template Parameter_Type<T> kernel_param, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		Kernel::apply(kernel_param, tid);
	}
}


template< typename T, typename Kernel, typename Launch_Config>
void cuda_loop(typename Kernel::template Parameter_Type<T>& kernel_param, int size)
{
	int tpb = Launch_Config::thread_per_block;
	int bpg = (size - 1) / tpb + 1;
	cuda_for_loop<T, Kernel > << <bpg, tpb >> > (kernel_param, size);
}

#define DELEARE_CUDA_LOOP_TEMPLATE_1(T,Kernel,Launch_Config,...)\
template void cuda_loop<T, Kernel<T,__VA_ARGS__>,Launch_Config>(typename Kernel<T,__VA_ARGS__>::template Parameter_Type<T>& kernel_param,int size );\

DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<16>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<32>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<64>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<128>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<256>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<512>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Add_Kernel,  Launch_Config<1024>)

DELEARE_CUDA_LOOP_TEMPLATE_1(float,  Mupltiply_Add_N_Times_Kernel,  Launch_Config<128>, Repeat<10>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float,  Mupltiply_Add_N_Times_Kernel,  Launch_Config<128>, Repeat<100>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float,  Mupltiply_Add_N_Times_Kernel,  Launch_Config<128>, Repeat<1000>)
DELEARE_CUDA_LOOP_TEMPLATE_1(double, Mupltiply_Add_N_Times_Kernel,  Launch_Config<128>, Repeat<100>)
DELEARE_CUDA_LOOP_TEMPLATE_1(double, Mupltiply_Add_N_Times_Kernel,  Launch_Config<128>, Repeat<1000>)

DELEARE_CUDA_LOOP_TEMPLATE_1(double, Convolution_Kernel,  Launch_Config<128>, Neighbor_Width<1>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float,  Convolution_Kernel,  Launch_Config<128>, Neighbor_Width<1>)
DELEARE_CUDA_LOOP_TEMPLATE_1(float,  Convolution_Kernel,  Launch_Config<128>, Neighbor_Width<2>)


DELEARE_CUDA_LOOP_TEMPLATE_1(float, Matrix_Vector_Multiplication_ELL,  Launch_Config<128> )
DELEARE_CUDA_LOOP_TEMPLATE_1(float, Matrix_Vector_Multiplication_CSR,  Launch_Config<128> )


