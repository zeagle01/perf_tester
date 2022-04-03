
#pragma once


#include "test_case.h"



class Serial_Quick_Sort :public Test_Case
{
public:
	void init(int size) override ;
	void run() override ;
	void sync_wait() override {};//for async test case
	size_t get_operation_size_with_respect_to_byte() override;
	size_t get_problem_size_in_byte() override;
	bool verify() override;
	std::string get_name() override { return "Serial_Quick_Sort"; };

private:
	std::vector<int> m_in;
	std::vector<int> m_out;
	int m_count;
	void quick_sort(std::vector<int>& a, int b, int e);
	void quick_sort(std::vector<int>& a, int& count, int b, int e);
};