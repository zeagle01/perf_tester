
#pragma  once


#include "cuda_runtime.h"

#ifdef __CUDACC__
#define KERNEL_MODIFIER __device__ inline
#else
#define KERNEL_MODIFIER  inline
#endif


template<typename T>
struct Matrix_In_Matrix_Out
{
	T* out_data;
	int out_row;
	int out_col;

	T* in_data;
	int in_row;
	int in_col;
};


template<template<typename >typename T>
struct With_Parameter_Type { template<typename U>using Parameter_Type = T<U>; };




/////////////////////////////////////////////////////
template<typename T>
struct Add_Kernel :With_Parameter_Type<Matrix_In_Matrix_Out>
{

	static KERNEL_MODIFIER
		void apply(Matrix_In_Matrix_Out<T>& param, int i)
	{
		T* in0 = &param.in_data[0 * param.out_col];
		T* in1 = &param.in_data[1 * param.out_col];
		param.out_data[i] = in0[i] + in1[i];
	}

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
struct Mupltiply_Add_N_Times_Kernel:With_Parameter_Type<Matrix_In_Matrix_Out>
{

	static KERNEL_MODIFIER
		void apply(Matrix_In_Matrix_Out<T>& param, int i)
	{
		T* in0 = &param.in_data[0 * param.out_col];
		T* in1 = &param.in_data[1 * param.out_col];
		T out_i = param.out_data[i];
		T in0_i = in0[i];
		T in1_i = in1[i];
		for (int j = 0; j < repeat_count::value; j++)
		{
			out_i = in0_i * out_i + in1_i;

		}
		param.out_data[i] = out_i;
	}

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



////////////////////////////

template<typename T,typename neighbor_width>
struct Convolution_Kernel :With_Parameter_Type<Matrix_In_Matrix_Out>
{

	static KERNEL_MODIFIER
		void apply(Matrix_In_Matrix_Out<T>& param, int i)
	{

		T cov = 0;
		for (int j = -neighbor_width::value; j <=neighbor_width::value; j++)
		{
			cov += param.in_data[neighbor_width::value + i + j];
		}
		cov /= (2 * neighbor_width::value + 1);
		param.out_data[i] = cov;

	}

	static KERNEL_MODIFIER void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row, int i)
	{
		T cov = 0;
		for (int j = -neighbor_width::value; j <=neighbor_width::value; j++)
		{
			cov += in[neighbor_width::value + i + j];
		}
		cov /= (2 * neighbor_width::value + 1);
		out[i] = cov;
	}
};
