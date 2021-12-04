
#pragma once

#include <vector>
#include "type_list.h"


template<
	typename T,
	template<typename U, template<typename> typename Kernel_Param, typename ...P >typename Device,
	typename Kernel,
	typename ... Device_Param
	>
struct Loop ;

template<typename T, template<typename> typename Kernel_Param>
struct Empty_Device
{
	using Data_Type = T;
	void dowload(Kernel_Param<T>& host_param, const Kernel_Param<T>& device_param) { }
	void upload(Kernel_Param<T>& device_param, const Kernel_Param<T>& host_param) { }
	void sync_wait() {}
	void free_device_source(Kernel_Param<T>& device_param) {}
};


template< 
	typename T,
	template<typename U, template<typename> typename Kernal_Param, typename ...P> typename Device,
	typename ...Device_Param
>
struct Loop_Locator
{
	template<typename Kernel >
	using Looper = Loop<T, Device, Kernel, Device_Param...>;
};

template<typename T, template<typename> typename Kernel_Param>
struct CPU_Base :Empty_Device<T, Kernel_Param>
{
	void upload(Kernel_Param<T>& device_param, const Kernel_Param<T>& host_param) 
	{
		device_param = host_param;
	}
};

/// CPU
template<typename T, template<typename> typename Kernel_Param>
struct CPU_Imp :CPU_Base<T, Kernel_Param>, Loop_Locator<T, CPU_Imp>
{
};

struct CPU 
{ 
	using param_list = type_list<>;

	template <typename T, template<typename> typename Kernel_Param>
	using type = CPU_Imp<T, Kernel_Param >;
};


/// OMP
template<typename T, template<typename> typename Kernel_Param>
struct OMP_Imp :CPU_Base<T, Kernel_Param>, Loop_Locator<T, OMP_Imp>
{
	void sync_wait()
	{
#pragma omp barrier
	}

};

struct OMP 
{ 
	using param_list = type_list<>;
	template <typename T, template<typename> typename Kernel_Param>
	using type = OMP_Imp<T, Kernel_Param >;
};

//PPL
template<typename T, template<typename> typename Kernel_Param>
struct PPL_Imp : CPU_Base<T, Kernel_Param>, Loop_Locator<T, PPL_Imp> { };

struct PPL 
{ 
	using param_list = type_list<>;
	template <typename T,template<typename> typename Kernel_Param>
	using type = PPL_Imp<T, Kernel_Param>;
};


