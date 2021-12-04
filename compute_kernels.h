
#pragma  once


#include "cuda_runtime.h"
#include <vector>

#ifdef __CUDACC__
#define KERNEL_MODIFIER __device__ inline
#else
#define KERNEL_MODIFIER  inline
#endif

template<typename T>
struct Device_Pointer
{
	T* data;
	int count;
	using data_type = T;

	KERNEL_MODIFIER T& operator[](int i) { return data[i]; }
	KERNEL_MODIFIER const T&  operator[](int i)const { return data[i]; }

	operator T* (){ return data; }
	operator T const  *() const { return data; }

	Device_Pointer& operator=(std::vector<T>& v)
	{
		data = v.data();
		count = v.size();
		return *this;
	}
};


template<typename T>
struct Matrix_In_Matrix_Out
{
	Device_Pointer<T> out_data;
	Device_Pointer<T> in_data;
	int out_row;
	int out_col;
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

};





///////////// CSR Matrix vector multiply
template<typename T>
struct CSR_Matrix_And_Vector
{
	Device_Pointer<T> A;
	Device_Pointer<int> I;
	Device_Pointer<int> J;
	Device_Pointer<float> b;

	Device_Pointer<T> x;

};

template<typename T>
struct Matrix_Vector_Multiplication_CSR :With_Parameter_Type<CSR_Matrix_And_Vector>
{

	static KERNEL_MODIFIER
		void apply(Matrix_In_Matrix_Out<T>& param, int i)
	{
		for (int k = param.I[i]; k <param. I[i + 1]; k++)
		{
			int j = param.J[k];
			param.x[i] += param.A[k] * param.b[j];
		}
	}

};

///////////// CSR Matrix vector multiply
template<typename T>
struct ELL_Matrix_And_Vector
{
	Device_Pointer<T> A;
	Device_Pointer<int> J;
	int col;

	Device_Pointer<T> b;

	Device_Pointer<T> x;

};

template<typename T>
struct Matrix_Vector_Multiplication_ELL :With_Parameter_Type<ELL_Matrix_And_Vector>
{

	static KERNEL_MODIFIER
		void apply(ELL_Matrix_And_Vector<T>& param, int i)
	{
		//int col
		//for (int k = 0; k < param.J[i]; k++)
		//{
		//	param.A[]
		//}
	}

};
