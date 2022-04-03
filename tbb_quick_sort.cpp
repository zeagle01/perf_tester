
#include "tbb_quick_sort.h"
#include <random>
#include <numeric>
#include <algorithm>
#include "tbb/tbb.h"


void TBB_Quick_Sort::init(int size)
{
	m_in.resize(size);
	m_out.resize(size);
	std::iota(m_in.begin(), m_in.end(), 0);
	std::shuffle(m_in.begin(), m_in.end(), std::default_random_engine{0});
	m_out = m_in;
	m_count = 0;
}

void TBB_Quick_Sort::run()
{
	quick_sort_parallel(m_out, 0, m_out.size(), 512);
}

size_t TBB_Quick_Sort::get_operation_size_with_respect_to_byte()
{
	m_count = 0;
	auto temp = m_in;
	quick_sort_serial(temp, m_count, 0, temp.size());
	return m_count * sizeof(int);
}

size_t TBB_Quick_Sort::get_problem_size_in_byte()
{
	return m_in.size() * sizeof(int);
}

bool TBB_Quick_Sort::verify()
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

void TBB_Quick_Sort::quick_sort_serial(std::vector<int>& a, int b, int e)
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

		quick_sort_serial(a, b, s);
		quick_sort_serial(a, s + 1, e);
	}

}

void TBB_Quick_Sort::quick_sort_parallel(std::vector<int>& a, int b, int e,int cutoff)
{
	if (e - b < cutoff)
	{
		quick_sort_serial(a, b, e);

	}
	else
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

		tbb::parallel_invoke(
			[=, this, &a]() {quick_sort_parallel(a, b, s, cutoff); },
			[=, this, &a]() {quick_sort_parallel(a, s + 1, e, cutoff); }
		);
	}
}

void TBB_Quick_Sort::quick_sort_serial(std::vector<int>& a, int& count, int b, int e)
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
		quick_sort_serial(a, count, b, s);
		quick_sort_serial(a, count, s + 1, e);
	}

}




void TBB_Parallel_Sort::run()
{
	tbb::parallel_sort(m_out.begin(), m_out.end());
}
	
