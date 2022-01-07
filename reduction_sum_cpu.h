

#pragma once

#include "test_case.h"


class Reduction_Sum_CPU :public Test_Case
{
public:
	void init(int size) override ;
	void run() override ;
	void sync_wait() override {};//for async test case
	size_t get_operation_size_with_respect_to_byte() override { return  m_size / sizeof(int); }
	size_t get_problem_size_in_byte() override { return m_size * sizeof(int); }
	bool verify() override;
	std::string get_name()override { return "Reduction_Sum_CPU"; };

private:
	int m_size;
	std::vector<int> m_data;
	int m_sum;
};

