
#pragma once

#include <vector>
#include "type_list.h"


template<
	typename T,
	template<typename U, typename ...P >typename Device,
	typename Kernel,
	typename ... Device_Param
	>
struct Loop ;

template<typename T>
struct Empty_Device
{
	using Data_Type = T;
	void dowload(std::vector<T>& host_out, T* device_out) { }
	void upload(T*& m_device_in, T*& m_device_out, const std::vector<T>& m_in, const std::vector<T>& m_out) { }
	void sync_wait() {}
	void free_device_source(T* device_out, T* device_in) {}
};


template< 
	typename T,
	template<typename U, typename ...P> typename Device,
	typename ...Device_Param
>
struct Loop_Locator
{
	template<typename Kernel>
	using Looper = Loop<T, Device, Kernel, Device_Param...>;
};

template<typename T>
struct CPU_Base :Empty_Device<T>
{
	void upload(T*& m_device_in, T*& m_device_out, const std::vector<T>& m_in, const std::vector<T>& m_out)
	{
		m_device_in = const_cast<T*>(&m_in[0]);
		m_device_out = const_cast<T*>(&m_out[0]);
	}
};

/// CPU
template<typename T>
struct CPU_Imp :CPU_Base<T>, Loop_Locator<T, CPU_Imp>
{

	void upload(T*& m_device_in, T*& m_device_out, const std::vector<T>& m_in, const std::vector<T>& m_out)
	{
		m_device_in = const_cast<T*>(&m_in[0]);
		m_device_out = const_cast<T*>(&m_out[0]);
	}
};

struct CPU 
{ 
	using param_list = type_list<>;

	template <typename T>
	using type = CPU_Imp<T >;
};


/// OMP
template<typename T>
struct OMP_Imp :CPU_Base<T>, Loop_Locator<T, OMP_Imp>
{
	void sync_wait()
	{
#pragma omp barrier
	}

};

struct OMP 
{ 
	using param_list = type_list<>;
	template <typename T>
	using type = OMP_Imp<T >;
};

//PPL
template<typename T>
struct PPL_Imp : CPU_Base<T>, Loop_Locator<T, PPL_Imp> { };

struct PPL 
{ 
	using param_list = type_list<>;
	template <typename T>
	using type = PPL_Imp<T >;
};


//////////////////cuda////////////////
template<int N>
struct Launch_Config
{
	static constexpr int thread_per_block = N;
};

template<typename T, typename Launch_Config>
struct CUDA_Imp :Loop_Locator<T, CUDA_Imp, Launch_Config>
{
	void dowload(std::vector<T>& host_out, T* device_out)
	{

		cudaMemcpy(host_out.data(), device_out, sizeof(T) * host_out.size(), cudaMemcpyDeviceToHost);
	}


	void upload(T*& device_in, T*& device_out, const std::vector<T>& in, const std::vector<T>& out)
	{

		cudaMalloc(&device_in, sizeof(T) * in.size());
		cudaMalloc(&device_out, sizeof(T) * out.size());

		cudaMemcpy(device_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(device_in, in.data(), sizeof(T) * in.size(), cudaMemcpyHostToDevice);
	}
	void sync_wait()
	{
		cudaDeviceSynchronize();
	}

	void free_device_source(T* device_out, T* device_in)
	{
		cudaFree(device_out);
		cudaFree(device_in);
	}
};

template< typename...P>
struct CUDA 
{
	using param_list = type_list<P...>;
	template <typename T>
	using type = CUDA_Imp<T, P... >;
};

