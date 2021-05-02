
#pragma once

#include "devices.h"
#include "ppl.h"



template<typename T>
struct Default_Loop
{

	void init_extra_data(const std::vector<T>& in, int in_col, int in_row, int out_col, int out_row) { }

	void apply(T* in, T* out, void* extra_data, int in_col, int in_row, int out_col, int out_row) { }

};





/// serial cpu ///////////////
template<
	typename T,
	template<typename U>typename Device,
	template<typename U>typename Kernel,
	typename ... Param
	>
struct Loop { };


template<
	typename T,
	template<typename U>typename Kernel
	>
struct Loop<T, CPU, Kernel> :Default_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
		for (int i = 0; i < size; i++)
		{
			Kernel<T>::apply(in, out, in_col, in_row, out_col, out_row, i);
		}
	}
};


/// omp ///////////////
template<
	typename T,
	template<typename U>typename Kernel
	>
struct Loop<T, OMP, Kernel> :Default_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			Kernel<T>::apply(in, out, in_col, in_row, out_col, out_row, i);
		}
	}
};

/// ppl ///////////////

template<typename T>
struct PPL_Loop 
{
	void init_extra_data(const std::vector<T>& in, int in_col, int in_row, int out_col, int out_row)
	{
		m_iterative_index.resize(out_col);
		for (int i = 0; i < out_col; i++)
		{
			m_iterative_index[i] = i;
		}
	}

	std::vector<int> m_iterative_index;
};

template<
	typename T,
	template<typename U>typename Kernel
	>
struct Loop<T, PPL, Kernel> :public PPL_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
		auto fn = [&](int i)
		{
			Kernel<T>::apply(in, out, in_col, in_row, out_col, out_row, i);
		};

		Concurrency::parallel_for_each(m_iterative_index.begin(), m_iterative_index.end(), fn);
	}
};

/// cuda ///////////////
template< typename T, typename Kernel, int thread_per_block>
void cuda_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row);

template<int N>
struct CUDA_1d_Launch_Config
{
	static constexpr int thread_per_block = N;
};

template<
	typename T,
	template<typename U>typename Kernel,
	typename Launch_Config
	>
	struct Loop<T, CUDA, Kernel, Launch_Config> :Default_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
		cuda_loop<T, Kernel<T>, Launch_Config::thread_per_block>(in, out, size, in_col, in_row, out_col, out_row);
	}
};


