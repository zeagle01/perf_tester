#include "vector_add_omp.h"


void Omp_Vector_Add::init(int size)
{
	m_size = size;
	m_a.resize(size);
	m_b.resize(size);
	m_c.resize(size);

	std::fill(m_a.begin(), m_a.end(), 1.f);
	std::fill(m_b.begin(), m_b.end(), 1.f);
	std::fill(m_c.begin(), m_c.end(), 0.f);

}
void Omp_Vector_Add::run()
{
#pragma omp parallel for
	for (int i = 0; i < m_c.size(); i++)
	{
		m_c[i] = m_a[i] + m_b[i];
	}
}
void Omp_Vector_Add::sync_wait()
{
#pragma omp barrier
}

size_t Omp_Vector_Add::get_size_in_byte()
{
	return (m_size) * sizeof(float);

}

bool Omp_Vector_Add::verify()
{

	for (int i = 0; i < m_size; i++)
	{
		if (m_c[i] != m_a[i] + m_b[i])
		{
			return false;
			break;
		}
	}
	return true;

}
