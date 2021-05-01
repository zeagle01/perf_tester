
#pragma  once


#include "cuda_runtime.h"
#include "vector_add_cuda.h"
#include "ppl.h"


template<typename T>
struct Default_Kernel
{

	void init_extra_data(const std::vector<T>& in, int in_col, int in_row, int out_col, int out_row) { }

	void apply(T* in, T* out, void* extra_data, int in_col, int in_row, int out_col, int out_row) { }

};



template<typename T>
struct CPU_Add :public Default_Kernel<T>
{

	void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row)
	{

		T* in0 = &in[0 * out_col];
		T* in1 = &in[1 * out_col];
		for (int i = 0; i < out_col; i++)
		{
			out[i] = in0[i] + in1[i];
		}
	}

};


template<typename T>
struct OMP_Add :Default_Kernel<T>
{
	void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row)
	{

		T* in0 = &in[0 * out_col];
		T* in1 = &in[1 * out_col];

#pragma omp parallel for
		for (int i = 0; i < out_col; i++)
		{
			out[i] = in0[i] + in1[i];
		}
	}

};



template<typename T>
struct CUDA_Add :Default_Kernel<T>
{
	void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row);

};


template<typename T>
struct PPL_Add 
{

	void init_extra_data(const std::vector<T>& in, int in_col, int in_row, int out_col, int out_row)
	{
		m_iterative_index.resize(out_col);
		for (int i = 0; i < out_col; i++)
		{
			m_iterative_index[i] = i;
		}
	}

	void apply(T* in, T* out, int in_col, int in_row, int out_col, int out_row)
	{
		auto fn = [&](int i)
		{
			T* in0 = &in[0];
			T* in1 = &in[out_col];
			out[i] = in0[i] + in1[i];
		};
		Concurrency::parallel_for_each(m_iterative_index.begin(), m_iterative_index.end(), fn);
	}

	std::vector<int> m_iterative_index;

};
