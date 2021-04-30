

#include "test_case.h"
#include <vector>


class Vector_Add_Ppl :public Test_Case
{

public:
		void init(int size)override ;
		void run() override;
		void sync_wait()override ;//for async test case
		size_t get_size_in_byte() override;
		bool verify() override ;
		std::string get_name() override { return "Vector_Add_Ppl"; };

private:
	std::vector<float> m_a;
	std::vector<float> m_b;
	std::vector<std::pair<int, float>> m_c;
	int m_size;
};

class Vector_Add_Multiply_Ppl :public Test_Case
{

public:
		void init(int size)override ;
		void run() override;
		void sync_wait()override ;//for async test case
		size_t get_size_in_byte() override;
		bool verify() override ;
		std::string get_name() override { return "Vector_Add_Multiply_Ppl"; };

private:
	int m_compute_intensity = 100;
	float expect_value = 10.f;
	std::vector<float> m_a;
	std::vector<float> m_b;
	std::vector<std::pair<int, float>> m_c;
	int m_size;
};
