
#pragma once

#include "devices.h"
#include "ppl.h"



template<typename T, template <typename> typename Kernel_Param>
struct Default_Loop
{

	void before_loop(const Kernel_Param<T>& kernel_param_host, const Kernel_Param<T>& kernel_param_device, int size) { }

	void apply(Kernel_Param<T>& kernel_param_device, int size) { }

};





/// serial cpu ///////////////

template<
	typename T,
	template<typename U, template<typename> typename Kernel_Param, typename ...P >typename Device,
	typename Kernel,
	typename ... Device_Param
	>
	struct Loop {};


template<
	typename T,
	typename Kernel
	>
	struct Loop<T, CPU_Imp, Kernel> :Default_Loop<T, Kernel::Parameter_Type>
{

	void apply(typename Kernel::template Parameter_Type<T>& kernel_param_device, int size)
	{
		for (int i = 0; i < size; i++)
		{
			Kernel::apply(kernel_param_device, i);
		}
	}

};


/// omp ///////////////
template<
	typename T,
	typename Kernel
	>
struct Loop<T, OMP_Imp, Kernel> :Default_Loop<T, Kernel::template Parameter_Type >
{

	void apply(Kernel::template Parameter_Type& kernel_param_device, int size)
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
struct Loop<T,  PPL_Imp, Kernel>
{

	void before_loop(const typename Kernel::template Parameter_Type<T>& kernel_param_host, const typename Kernel::template Parameter_Type<T>& kernel_param_device, int size)
	{ 
		m_iterative_index.resize(size);
		for (int i = 0; i < size; i++)
		{
			m_iterative_index[i] = i;
		}
	}

	std::vector<int> m_iterative_index;
};

/// cuda ///////////////
template< typename T, typename Kernel ,typename Launch_Config>
void cuda_loop(T* in, T* out, int size, int in_col, int in_row, int out_col, int out_row);

template< typename T, typename Kernel, typename Launch_Config>
void cuda_loop(typename Kernel::template Parameter_Type<T>& kernel_param, int size);


template<
	typename T,
	typename Kernel,
	typename Launch_Config
	>
struct Loop<T, CUDA_Imp, Kernel, Launch_Config> :Default_Loop<T, Kernel::template Parameter_Type>
{

	void apply(typename Kernel::template Parameter_Type<T>& kernel_param_device, int size)
	{
		cuda_loop<T, Kernel, Launch_Config>(kernel_param_device, size);
	}
};


