#include "vector_add_ppl.h"
#include "ppl.h"


void Vector_Add_Ppl::init(int size)
{
	m_size = size;
	m_a.resize(size);
	m_b.resize(size);
	m_c.resize(size);

	std::fill(m_a.begin(), m_a.end(), 1.f);
	std::fill(m_b.begin(), m_b.end(), 1.f);
	for (int i = 0; i < m_c.size(); i++)
	{
		m_c[i].first = i;
		m_c[i].second = 0.f;
	}

}
void Vector_Add_Ppl::run()
{
	//for (int i = 0; i < m_c.size(); i++)
	auto fn=[&](std::pair<int,float>& ii)
	{
		ii.second = m_b[ii.first] + m_a[ii.first];
	};
	Concurrency::parallel_for_each(m_c.begin(), m_c.end(), fn);
}
void Vector_Add_Ppl::sync_wait()
{
}

size_t Vector_Add_Ppl::get_size_in_byte()
{
	return (m_size) * sizeof(float);

}

bool Vector_Add_Ppl::verify()
{

	for (int i = 0; i < m_size; i++)
	{
		if (m_c[i].second != m_a[i] + m_b[i])
		{
			return false;
			break;
		}
	}
	return true;

}

////////////////////////////////////

void Vector_Add_Multiply_Ppl::init(int size)
{
	m_size = size;
	m_a.resize(size);
	m_b.resize(size);
	m_c.resize(size);

	std::fill(m_a.begin(), m_a.end(), 0.9f);
	std::fill(m_b.begin(), m_b.end(), 1.f);
	for (int i = 0; i < m_c.size(); i++)
	{
		m_c[i].first = i;
		m_c[i].second = 0.f;
	}

}
void Vector_Add_Multiply_Ppl::run()
{
	//for (int i = 0; i < m_c.size(); i++)
	auto fn=[&](std::pair<int,float>& c)
	{
		int i = c.first;
		float cc = c.second;
		for (int j = 0; j < m_compute_intensity; j++)
		{
			cc = cc * m_a[i] + m_b[i];
		}
		c.second = cc;
	};
	Concurrency::parallel_for_each(m_c.begin(), m_c.end(), fn);
}
void Vector_Add_Multiply_Ppl::sync_wait()
{
}

size_t Vector_Add_Multiply_Ppl::get_size_in_byte()
{
	return (m_size) * sizeof(float) * m_compute_intensity;
}

bool Vector_Add_Multiply_Ppl::verify()
{

	for (int i = 0; i < m_size; i++)
	{
		if (std::abs(m_c[i].second - expect_value) > 1e-2f)
		{
			return false;
			break;
		}
	}
	return true;

}
