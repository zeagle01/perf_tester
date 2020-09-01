



#include "test_case.h"

#include <vector>

#include "curand.h" //the cuda random number lib


class Cuda_Random:public Test_Case
{

public:
	~Cuda_Random();
		void init(int size)override ;
		void run() override;
		void sync_wait()override ;//for async test case
		size_t get_size_in_byte() override;
		bool verify() override ;
		std::string get_name() override{ return "Cuda_Random"; };

private:

	void free_cuda();

	std::vector<float> m_host_data;


	float* m_device_data = nullptr;

	int m_size;

	curandGenerator_t m_gen;
};

