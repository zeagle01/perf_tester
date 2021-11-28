

#include "runner.h"

#include "log.h"
#include "profiler.h"
#include "cuda_random.h"
#include "composed_test_case.h"
#include "devices.h"
#include "problems.h"
#include "compute_kernels.h"


void Runner::run(Test_Case* test_case)
{
	RECORD_FUNCTION_DURATION_RETURN(&m_duration);
	//RECORD_FUNCTION_DURATION();

	test_case->run();
	test_case->sync_wait();

}



void Runner::run()
{
	if (!os)
	{
		os = std::make_unique<std::ofstream>("result.txt");
	}

	m_cases.push_back(std::make_unique<
		Composed_Test_Case<   Convolution<float, Neighbor_Width<1>>, CPU >
	>());

	m_cases.push_back(std::make_unique<
		Composed_Test_Case<  Multiply_Add_N_Times<float, Repeat<100>>, CPU >
	>());


	m_cases.push_back(std::make_unique<
		Composed_Test_Case<   Convolution<float, Neighbor_Width<1>>, CUDA<Launch_Config<128>> >
	>());

	m_cases.push_back(std::make_unique<
		Composed_Test_Case<  Multiply_Add_N_Times<float, Repeat<100>>, CUDA<Launch_Config<128>> >
	>());




//	///////////cuda launch config test/////////////

	//m_cases.push_back(std::make_unique<Cuda_Random>());


	*os << "#";
	for (int i = 0; i < m_cases.size(); i++)
	{
		*os << " " << m_cases[i]->get_name();
	}
	*os << std::endl;

	for (int size_i = m_min_size_in_log2; size_i < m_max_size_in_log2; size_i++)
	{
		int size = 1 << size_i;

		CE_INFO("----------size {0}", size);

		for (int i = 0; i < m_cases.size(); i++)
		{
			m_cases[i]->init(size);

			long average_duration=0;
			for (int avg_i = 0; avg_i < m_average_num; avg_i++)
			{
				run(m_cases[i].get());
				average_duration += m_duration;
				CE_INFO("{0} duration {1} ", m_cases[i]->get_name(), m_duration);
			}
			average_duration /= m_average_num;
			*os << m_cases[i]->get_problem_size_in_byte() << " ";
			*os << m_cases[i]->get_operation_size_with_respect_to_byte() << " ";
			*os << average_duration << " ";

			if (m_verify)
			{
				bool good = m_cases[i]->verify();
				if (!good)
				{
					CE_ERROR("case {0} {1} verify failed", m_cases[i]->get_name(), (void*)m_cases[i].get());
				}
			}
		}
		*os << std::endl;

	}


}

