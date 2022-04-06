
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

	std::vector<std::unique_ptr<Test_Case>> get_cases();
	void run(Test_Case* test_case);

	//std::vector<std::unique_ptr<Test_Case>> m_cases;
	long m_duration;

	int m_average_num = 1;
	int m_max_size_in_log2 = 22;
	int m_min_size_in_log2 = 8;
//	int m_max_size_in_log2 = 5;
//	int m_min_size_in_log2 = 1;

	bool m_verify = true;

	std::unique_ptr<std::ofstream >os;



};
