
#pragma once

#include <string>



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



template<typename T>
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

	virtual size_t get_size_in_byte() override { return m_size*sizeof(T); }

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
	std::vector<T> m_v0;
	std::vector<T> m_v1;
	std::vector<T> m_result;
};


template<>
std::string Vector_Add<int>::get_name() { return "Vector_Add<int>"; }

template<>
std::string Vector_Add<float>::get_name() { return "Vector_Add<float>"; }