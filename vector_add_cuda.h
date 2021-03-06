

#include "test_case.h"
#include <vector>


class Cuda_Vector_Add:public Test_Case
{

public:
	~Cuda_Vector_Add();
		void init(int size)override ;
		void run() override;
		void sync_wait()override ;//for async test case
		size_t get_size_in_byte() override;
		bool verify() override ;
		std::string get_name() override{ return "Cuda_Vector_Add"; };

private:
	std::vector<float> m_a;
	std::vector<float> m_b;
	std::vector<float> m_c;

	void free_cuda();

	float* da = nullptr;
	float* db = nullptr;
	float* dc = nullptr;
	int m_size;
};


