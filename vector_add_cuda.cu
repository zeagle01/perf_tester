



#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "vector_add_cuda.h"

__global__ void vector_add(int num,float* c, float* a, float* b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num)
	{
		c[tid] = a[tid] + b[tid];
	}
}

void Cuda_Vector_Add::init(int size)
{
	m_size = size;
	m_a.resize(size);
	m_b.resize(size);
	m_c.resize(size);

	int size_in_byte = size * sizeof(float);

	free_cuda();
	cudaMalloc(&da, size_in_byte);
	cudaMalloc(&db, size_in_byte);
	cudaMalloc(&dc, size_in_byte);
	for (int i = 0; i < m_a.size(); i++)
	{
		m_a[i] = 1.f;
		m_b[i] = 1.f;
		m_c[i] = 0.f;
	}

	cudaMemcpy(da, m_a.data(), size_in_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(db, m_b.data(), size_in_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(dc, m_c.data(), size_in_byte, cudaMemcpyHostToDevice);
}

void Cuda_Vector_Add::run() 
{
	int tpb = 128;
	int bpg = (m_size - 1) / tpb + 1;
	vector_add << <bpg, tpb >> > (m_size, dc, da, db);
}

void Cuda_Vector_Add::sync_wait()
{
	cudaDeviceSynchronize();
};

size_t Cuda_Vector_Add::get_size_in_byte()
{
	return m_size * sizeof(float);
}

bool Cuda_Vector_Add::verify() 
{


	cudaMemcpy(m_a.data(), da, get_size_in_byte(), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_b.data(), db, get_size_in_byte(), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_c.data(), dc, get_size_in_byte(), cudaMemcpyDeviceToHost);

	for (int i = 0; i < m_size; i++)
	{
		if (m_c[i] != m_a[i] + m_b[i])
		{
			return false;
			break;
		}
	}
	return true;
};

Cuda_Vector_Add::~Cuda_Vector_Add()
{
	free_cuda();
}

void Cuda_Vector_Add::free_cuda()
{
	if (da)
	{

		cudaFree(da);
	}
	if (db)
	{

		cudaFree(db);
	}
	if (dc)
	{
		cudaFree(dc);
	}

}

/////////////////////////////////////////
__global__ void vector_add_multiply(int num, float* c, float* a, float* b,int compute_intensity)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num)
	{
		float cc = c[tid];
		for (int i = 0; i < compute_intensity; i++)
		{
			cc = cc * a[tid] + b[tid];
		}
		c[tid] = cc;
	}
}

void Cuda_Vector_Add_Multiply::init(int size)
{
	m_size = size;
	m_a.resize(size);
	m_b.resize(size);
	m_c.resize(size);

	int size_in_byte = size * sizeof(float);

	free_cuda();
	cudaMalloc(&da, size_in_byte);
	cudaMalloc(&db, size_in_byte);
	cudaMalloc(&dc, size_in_byte);
	for (int i = 0; i < m_a.size(); i++)
	{
		m_a[i] = 0.9f;
		m_b[i] = 1.f;
		m_c[i] = 0.f;
	}

	cudaMemcpy(da, m_a.data(), size_in_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(db, m_b.data(), size_in_byte, cudaMemcpyHostToDevice);
	cudaMemcpy(dc, m_c.data(), size_in_byte, cudaMemcpyHostToDevice);
}

void Cuda_Vector_Add_Multiply::run() 
{
	int tpb = 128;
	int bpg = (m_size - 1) / tpb + 1;
	vector_add_multiply << <bpg, tpb >> > (m_size, dc, da, db, m_compute_intensity);
}

void Cuda_Vector_Add_Multiply::sync_wait()
{
	cudaDeviceSynchronize();
};

size_t Cuda_Vector_Add_Multiply::get_size_in_byte()
{
	return m_size * sizeof(float) * m_compute_intensity;
}

bool Cuda_Vector_Add_Multiply::verify() 
{


	auto data_in_byte = m_size * sizeof(float);
	cudaMemcpy(m_a.data(), da, data_in_byte, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_b.data(), db, data_in_byte, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_c.data(), dc, data_in_byte, cudaMemcpyDeviceToHost);

	for (int i = 0; i < m_size; i++)
	{
		if (std::abs(m_c[i] - expect_value) > 1e-3f)
		{
			return false;
			break;
		}
	}
	return true;
};

Cuda_Vector_Add_Multiply::~Cuda_Vector_Add_Multiply()
{
	free_cuda();
}

void Cuda_Vector_Add_Multiply::free_cuda()
{
	if (da)
	{

		cudaFree(da);
	}
	if (db)
	{

		cudaFree(db);
	}
	if (dc)
	{
		cudaFree(dc);
	}

}
