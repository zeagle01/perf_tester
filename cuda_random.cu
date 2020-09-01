
#include "device_launch_parameters.h"
#include "cuda_runtime.h"


#include "cuda_random.h"



#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    std::getchar() ;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    std::getchar() ;}} while(0)

Cuda_Random::~Cuda_Random()
{
	free_cuda();
}

void Cuda_Random::init(int size)
{
	m_size = size;

	m_host_data.resize(size);

	CUDA_CALL(
	cudaMalloc(&m_device_data, sizeof(float) * size)
	);

	CURAND_CALL(
	curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT)
	);

	CURAND_CALL(
		curandSetPseudoRandomGeneratorSeed(m_gen, 0ull)
	);

}
void Cuda_Random::run()
{
	curandGenerateUniform(m_gen, m_device_data, m_size);

}

void Cuda_Random::sync_wait()
{

	cudaDeviceSynchronize();

}
size_t Cuda_Random::get_size_in_byte()
{
	return m_size * sizeof(float);
}

bool Cuda_Random::verify()
{

	cudaMemcpy(m_host_data.data(), m_device_data, get_size_in_byte(), cudaMemcpyDeviceToHost);

	for (int i=1;i<m_host_data.size();i++)
	{
		if (m_host_data[i] == m_host_data[i - 1])
		{
			return false;
		}
	}

	return true;
}

void Cuda_Random::free_cuda()
{
	if (m_device_data)
	{
		cudaFree(m_device_data);
	}

}
