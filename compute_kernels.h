
#pragma  once


#include "cuda_runtime.h"

#ifdef __CUDACC__
#define KERNEL_MODIFIER __device__ inline
#else
#define KERNEL_MODIFIER  inline
#endif



template<typename T>
struct Add_Kernel
{
	static KERNEL_MODIFIER
		void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row, int i)
	{
		T* in0 = &in[0 * out_col];
		T* in1 = &in[1 * out_col];
		out[i] = in0[i] + in1[i];
	}
};



////////////////////////////

template<typename T,typename repeat_count>
struct Mupltiply_Add_N_Times_Kernel
{

	static KERNEL_MODIFIER void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row, int i)
	{
		T* in0 = &in[0 * out_col];
		T* in1 = &in[1 * out_col];
		T out_i = out[i];
		T in0_i = in0[i];
		T in1_i = in1[i];
		for (int j = 0; j < repeat_count::value; j++)
		{
			out_i = in0_i * out_i + in1_i;

		}
		out[i] = out_i;
	}
};


