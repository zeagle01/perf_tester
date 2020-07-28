

#include "runner.h"

#include "log.h"
#include "profiler.h"


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

	m_cases.push_back(std::make_unique<Vector_Add<int>>());
	m_cases.push_back(std::make_unique<Vector_Add<float>>());


	*os << "#";
	for (int i = 0; i < m_cases.size(); i++)
	{
		*os << " " << m_cases[i]->get_name();
	}
	*os << std::endl;

	for (int size_i = m_min_size_in_log2; size_i < m_max_size_in_log2; size_i++)
	{
		int size = 1 << size_i;

		for (int i = 0; i < m_cases.size(); i++)
		{
			m_cases[i]->init(size);

			long average_duration=0;
			for (int avg_i = 0; avg_i < m_average_num; avg_i++)
			{
				run(m_cases[i].get());
				average_duration += m_duration;
				CE_INFO("duration {0}", m_duration);
			}
			average_duration /= m_average_num;
			*os << m_cases[i]->get_size_in_byte() << " ";
			*os << average_duration << " ";

			if (m_verify)
			{
				bool good = m_cases[i]->verify();
				if (!good)
				{
					CE_ERROR("case {0} verify failed", (void*)m_cases[i].get());
				}
			}
		}
		*os << std::endl;

	}


}

