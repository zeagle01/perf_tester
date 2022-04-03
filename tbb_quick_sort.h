
#pragma once

#include "test_case.h"

class TBB_Quick_Sort :public Test_Case
{
public:
	void init(int size) override ;
	void run() override ;
	void sync_wait() override {};//for async test case
	size_t get_operation_size_with_respect_to_byte() override;
	size_t get_problem_size_in_byte() override;
	bool verify() override;
	std::string get_name() override { return "TBB_Quick_Sort"; };

protected:
	std::vector<int> m_out;

private:
	std::vector<int> m_in;
	int m_count;

	void quick_sort_parallel(std::vector<int>& a, int b, int e,int cutoff);
	void quick_sort_serial(std::vector<int>& a, int b, int e);
	void quick_sort_serial(std::vector<int>& a, int& count, int b, int e);
};

class TBB_Parallel_Sort :public TBB_Quick_Sort
{
public:
	void run() override ;
	std::string get_name() override { return "TBB_Parallel_Sort"; };

};
