
#pragma once

#include <string>
#include <vector>







class Test_Case
{
	public:
		virtual void init(int size) {};
		virtual void run() {};
		virtual void sync_wait() {};//for async test case
		virtual size_t get_size_in_byte() { return 0; }
		virtual bool verify() { return false; };
		virtual std::string get_name() { return "test_case"; };

};



template<typename ove>
class Vector_Add :public Test_Case
{
public:
	std::string get_name() { return "Vector<T>"; }

	virtual void init(int size) 
	{
		m_size = size; 
		m_v0.resize(size);
		m_v1.resize(size);
		for (int i = 0; i < m_size; i++)
		{
			m_v0[i] = 1;
			m_v1[i] = 1;
		}

		m_result.resize(size);
	}

	virtual size_t get_size_in_byte() override { return m_size*sizeof(ove); }

	virtual void run() 
	{
		for (int i = 0; i < m_size; i++)
		{
			m_result[i] = m_v0[i] + m_v1[i];
		}
	};

	virtual bool verify() 
	{
		for (int i = 0; i < m_size; i++)
		{
			if (m_result[i] != m_v0[i] + m_v1[i])
			{
				return false;
				break;
			}
		}
		return true;
	};
private:
	int m_size = 0;
	std::vector<ove> m_v0;
	std::vector<ove> m_v1;
	std::vector<ove> m_result;
};


template<>
std::string Vector_Add<int>::get_name() { return "Vector_Add<int>"; }

template<>
std::string Vector_Add<float>::get_name() { return "Vector_Add<float>"; }





/////////////////////////
class Vector_Add_Multiply :public Test_Case
{
public:
	std::string get_name() { return "Vector_Add_Multiply"; }

	virtual void init(int size) 
	{
		m_size = size; 
		m_v0.resize(size);
		m_v1.resize(size);
		for (int i = 0; i < m_size; i++)
		{
			m_v0[i] = 0.9f;
			m_v1[i] = 1;
		}

		m_result.resize(size);
	}

	virtual size_t get_size_in_byte() override { return m_size * sizeof(float) * m_compute_intensity; }

	virtual void run() 
	{
		for (int i = 0; i < m_size; i++)
		{
			for (int j = 0; j < m_compute_intensity; j++)
			{
				m_result[i] = m_result[i] * m_v0[i] + m_v1[i];
			}
		}
	};

	virtual bool verify() 
	{
		for (int i = 0; i < m_size; i++)
		{
			if (std::abs(m_result[i] - expect_value) > 1e-2f)
			{
				return false;
				break;
			}
		}
		return true;
	};
private:
	int m_size = 0;
	float expect_value = 10.f;
	int m_compute_intensity = 100;
	std::vector<float> m_v0;
	std::vector<float> m_v1;
	std::vector<float> m_result;
};
