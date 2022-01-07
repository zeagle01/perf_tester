

#include "runner.h"

#include "log.h"
#include "profiler.h"
#include "cuda_random.h"
#include "composed_test_case.h"
#include "devices.h"
#include "problems.h"
#include "compute_kernels.h"
#include "launch_config.h"


void Runner::run(Test_Case* test_case)
{
	RECORD_FUNCTION_DURATION_RETURN(&m_duration);
	//RECORD_FUNCTION_DURATION();

	test_case->run();
	test_case->sync_wait();

}


std::vector<std::unique_ptr<Test_Case>> Runner::get_cases()
{
	std::vector<std::unique_ptr<Test_Case>> ret;

//	ret.push_back(std::make_unique<
//		Composed_Test_Case<   Vector_Add<float >, CPU >
//	>());
//
//	ret.push_back(std::make_unique<
//		Composed_Test_Case<   Convolution<float, Neighbor_Width<1>>, CPU >
//	>());
//
//	ret.push_back(std::make_unique<
//		Composed_Test_Case<  Multiply_Add_N_Times<float, Repeat<100>>, CPU >
//	>());
//
//	ret.push_back(std::make_unique<
//		Composed_Test_Case<   Vector_Add<float >, CUDA<Launch_Config<128>> >
//	>());
//
//	ret.push_back(std::make_unique<
//		Composed_Test_Case<   Convolution<float, Neighbor_Width<1>>, CUDA<Launch_Config<128>> >
//	>());
//
//	ret.push_back(std::make_unique<
//		Composed_Test_Case<  Multiply_Add_N_Times<float, Repeat<100>>, CUDA<Launch_Config<128>> >
//	>());

	ret.push_back(std::make_unique<
		Composed_Test_Case<  Laplician_1D<float, Matrix_Vector_Multiplication_ELL>, CUDA<Launch_Config<128>> >
	>());

	ret.push_back(std::make_unique<
		Composed_Test_Case<  Laplician_1D<float, Matrix_Vector_Multiplication_CSR>, CUDA<Launch_Config<128>> >
	>());

	ret.push_back(std::make_unique<
		Composed_Test_Case<  Laplician_1D<float, Matrix_Vector_Multiplication_ELL>, CPU >
	>());

	ret.push_back(std::make_unique<
		Composed_Test_Case<  Laplician_1D<float, Matrix_Vector_Multiplication_CSR>, CPU >
	>());




	//m_cases.push_back(std::make_unique<Cuda_Random>());

	return ret;
}

void Runner::run()
{
	if (!os)
	{
		os = std::make_unique<std::ofstream>("result.txt");
	}

	*os << "#";
	auto cases = get_cases();
	for (int i = 0; i < cases.size(); i++)
	{
		*os << " " << cases[i]->get_name();
	}
	*os << std::endl;

	for (int size_i = m_min_size_in_log2; size_i < m_max_size_in_log2; size_i++)
	{
		int size = 1 << size_i;

		CE_INFO("----------size {0}", size);

		auto cases = get_cases(); //recreate cases

		for (int i = 0; i < cases.size(); i++)
		{
			cases[i]->init(size);

			long average_duration=0;
			for (int avg_i = 0; avg_i < m_average_num; avg_i++)
			{
				run(cases[i].get());
				average_duration += m_duration;
				//CE_INFO("{0} duration {1} ", cases[i]->get_name(), m_duration);
				printf("%s duration %f \n", cases[i]->get_name().c_str(), m_duration);
			}
			average_duration /= m_average_num;
			*os << cases[i]->get_problem_size_in_byte() << " ";
			*os << cases[i]->get_operation_size_with_respect_to_byte() << " ";
			*os << average_duration << " ";

			if (m_verify)
			{
				bool good = cases[i]->verify();
				if (!good)
				{
					//CE_ERROR("case {0} {1} verify failed", cases[i]->get_name(), (void*)cases[i].get());
					printf("case %s %p verify failed \n", cases[i]->get_name().c_str(), (void*)cases[i].get());
				}
			}
		}
		*os << std::endl;

	}


}

