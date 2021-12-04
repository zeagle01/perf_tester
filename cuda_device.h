
#pragma once

#include "struct_helper.h"
#include "devices.h"

//////////////////cuda////////////////


struct Upload_Cuda_Data
{
	template<typename T >
	static void apply(T& p_out, const T& p_in)
	{
		p_out.count = p_in.count;
		auto size = p_out.count * sizeof(T::data_type);
		cudaMalloc(&p_out.data, p_out.count * sizeof(T::data_type));
		cudaMemcpy(p_out.data, p_in.data, size, cudaMemcpyHostToDevice);
	}
};

struct Download_Cuda_Data
{
	template<typename T >
	static void apply(T& p_out, const T& p_in)
	{
		auto size = p_out.count * sizeof(T::data_type);
		cudaMemcpy(p_out.data, p_in.data, size, cudaMemcpyDeviceToHost);
	}
};

struct Copy_NEWREFACTOR
{
	template<typename T >
	static void apply(T& v_out, const T& v_in)
	{
		v_out = v_in;
	}
};

struct Free_Cuda_Data
{
	template<typename T >
	static void apply(T& p_out)
	{
		cudaFree(p_out.data);
	}
};



template<typename T, template<typename> typename Kernel_Param, typename Launch_Config>
struct CUDA_Imp :Loop_Locator<T, CUDA_Imp, Launch_Config>
{

	void dowload(Kernel_Param<T>& host_param, const Kernel_Param<T>& device_param) 
	{ 
		For_Each_Member<Kernel_Param<T>>::apply<Dev_Ptr_Only, Download_Cuda_Data>(host_param, device_param);
	}

	void upload(Kernel_Param<T>& device_param, const Kernel_Param<T>& host_param) 
	{
		For_Each_Member<Kernel_Param<T>>::apply<Value_Only, Copy_NEWREFACTOR>(device_param, host_param);

		For_Each_Member<Kernel_Param<T>>::apply<Dev_Ptr_Only,Upload_Cuda_Data>(device_param, host_param);
	}


	void sync_wait()
	{
		cudaDeviceSynchronize();
	}


	void free_device_source(Kernel_Param<T>& device_param) 
	{
		For_Each_Member<Kernel_Param<T>>::apply<Dev_Ptr_Only, Free_Cuda_Data>(device_param);
	}
};

template< typename...P>
struct CUDA 
{
	using param_list = type_list<P...>;
	template <typename T,template<typename> typename Kernel_Param>
	using type = CUDA_Imp<T, Kernel_Param, P... >;
};
