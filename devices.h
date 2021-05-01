
#pragma once


template<typename T>
struct Default_Device
{
	void dowload(std::vector<T>& host_out, T* device_out) { }
	void upload(T*& m_device_in, T*& m_device_out, const std::vector<T>& m_in, const std::vector<T>& m_out) { }
	void sync_wait() {}
	void free_device_source(T* device_out, T* device_in) {}
};

template<typename T>
struct CPU :Default_Device<T>
{
	void upload(T*& m_device_in, T*& m_device_out, const std::vector<T>& m_in, const std::vector<T>& m_out)
	{
		m_device_in = const_cast<T*>(&m_in[0]);
		m_device_out = const_cast<T*>(&m_out[0]);
	}
};

template<typename T>
struct OMP: CPU<T>
{
	void sync_wait()
	{
#pragma omp barrier
	}

};

template<typename T>
struct CUDA
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

