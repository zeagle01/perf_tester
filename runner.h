
#pragma once


#include <memory> 
#include <vector>
#include "test_case.h"
#include <fstream>


class Test_Case;

class Runner
{
public:

	void run();

private:

	void run(Test_Case* test_case);

	std::vector<std::unique_ptr<Test_Case>> m_cases;
	//std::vector<long> m_durations;
	long m_duration;

	int m_average_num = 1;
	int m_max_size_in_log2=26;
	int m_min_size_in_log2=8;

	bool m_verify = true;

	std::unique_ptr<std::ofstream >os;



};
