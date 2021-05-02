
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
	template<typename U, typename ...P >typename Device,
	typename Kernel,
	typename ... Device_Param
	>
struct Loop { };


template<
	typename T,
	typename Kernel
	>
struct Loop<T, CPU_Imp, Kernel> :Default_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
		for (int i = 0; i < size; i++)
		{
			Kernel::apply(in, out, in_col, in_row, out_col, out_row, i);
		}
	}
};


/// omp ///////////////
template<
	typename T,
	typename Kernel
	>
struct Loop<T, OMP_Imp, Kernel> :Default_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			Kernel::apply(in, out, in_col, in_row, out_col, out_row, i);
		}
	}
};

/// ppl ///////////////
template<
	typename T,
	typename Kernel
	>
struct Loop<T, PPL_Imp, Kernel> 
{
	void init_extra_data(const std::vector<T>& in, int in_col, int in_row, int out_col, int out_row)
	{
		m_iterative_index.resize(out_col);
		for (int i = 0; i < out_col; i++)
		{
			m_iterative_index[i] = i;
		}
	}

	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
		auto fn = [&](int i)
		{
			Kernel::apply(in, out, in_col, in_row, out_col, out_row, i);
		};

		Concurrency::parallel_for_each(m_iterative_index.begin(), m_iterative_index.end(), fn);
	}

	std::vector<int> m_iterative_index;
};

/// cuda ///////////////
template< typename T, typename Kernel ,typename Launch_Config>
void cuda_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row);


template<
	typename T,
	typename Kernel,
	typename Launch_Config
	>
	struct Loop<T, CUDA_Imp, Kernel, Launch_Config> :Default_Loop<T>
{
	void apply(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row)
	{
		cuda_loop<T, Kernel, Launch_Config>(in, out, size, in_col, in_row, out_col, out_row);
	}
};


