
#include "serial_quick_sort.h"
#include <random>
#include <numeric>
#include <algorithm>


void Serial_Quick_Sort::init(int size)
{
	m_in.resize(size);
	m_out.resize(size);
	std::iota(m_in.begin(), m_in.end(), 0);
	std::shuffle(m_in.begin(), m_in.end(), std::default_random_engine{0});
	m_out = m_in;
	m_count = 0;
}

void Serial_Quick_Sort::run()
{
	quick_sort(m_out, 0, m_out.size());
}

size_t Serial_Quick_Sort::get_operation_size_with_respect_to_byte()
{
	m_count = 0;
	auto temp = m_in;
	quick_sort(temp, m_count, 0, temp.size());
	return m_count * sizeof(int);
}

size_t Serial_Quick_Sort::get_problem_size_in_byte()
{
	return m_in.size() * sizeof(int);
}

bool Serial_Quick_Sort::verify()
{
	for (int i = 0; i < m_out.size(); i++)
	{
		if (m_out[i] != i)
		{
			return false;

		}
	}
	return true;
}


void Serial_Quick_Sort::quick_sort(std::vector<int>& a, int b, int e)
{
	if (e - b > 1)
	{
		int i = b - 1;
		int r = e - 1;
		for (int j = b; j < e - 1; j++)
		{
			if (a[j] <= a[r])
			{
				i++;
				std::swap(a[i], a[j]);
			}
		}
		std::swap(a[i + 1], a[r]);


		int s = i + 1;
		quick_sort(a, b, s);
		quick_sort(a, s + 1, e);
	}
}

void Serial_Quick_Sort::quick_sort(std::vector<int>& a, int& count, int b, int e)
{

	if (e - b > 1)
	{
		int i = b - 1;
		int r = e - 1;
		for (int j = b; j < e - 1; j++)
		{
			count++;
			if (a[j] <= a[r])
			{
				i++;
				std::swap(a[i], a[j]);
			}
		}
		std::swap(a[i + 1], a[r]);


		int s = i + 1;
		quick_sort(a, count, b, s);
		quick_sort(a, count, s + 1, e);
	}

}
