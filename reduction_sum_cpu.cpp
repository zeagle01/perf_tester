

#include "reduction_sum_cpu.h" 


void Reduction_Sum_CPU::init(int size)
{
	m_size = size;
	m_data.assign(size, 1);
}

void Reduction_Sum_CPU::run()
{
	int sum = 0;
#pragma omp parallel for reduction(+: sum)
	for (int i = 0; i < m_size; i++)
	{
		sum += m_data[i];
	}

	m_sum = sum;
}

bool Reduction_Sum_CPU::verify()
{
	if (m_sum != m_size)
	{
		return false;
	}
	return true;
};
