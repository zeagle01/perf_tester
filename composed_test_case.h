
#pragma once

#include <sstream>

#include "loop.h"


template<
	typename T, 
	template<typename U> typename Problem, 
	template<typename U> typename Device, 
	template<typename U> typename Kernel 
	//template<typename U, typename De, typename Kernel> typename Updater
	>
class Composed_Test_Case :public Test_Case
{
public:
	void init(int size) override 
	{
		m_size = size;
		m_problem.init(m_in, m_out, m_in_col, m_in_row, m_out_col, m_out_row, size);
		m_device.upload(m_device_in, m_device_out, m_in, m_out);
		m_looper.init_extra_data(m_in, m_in_col, m_in_row, m_out_col, m_out_row);
	};
	void run() override 
	{ 
		//m_updater.apply(m_device_in, m_device_out, m_size, m_in_col, m_in_row, m_out_col, m_out_row);
		m_looper.apply(m_device_in, m_device_out, m_size, m_in_col, m_in_row, m_out_col, m_out_row);
	};

	//for async test case
	void sync_wait() override 
	{
		m_device.sync_wait(); 
	}

	size_t get_size_in_byte() override 
	{
		return m_problem.get_problem_size(m_size);
	}
	bool verify() override 
	{ 
		m_device.dowload(m_out, m_device_out);
		auto ret = m_problem.verify(m_in, m_out);
		m_device.free_device_source(m_device_out, m_device_in);
		return ret;
	};
	std::string get_name() override 
	{ 

		auto type_str = std::string(typeid(Kernel<T>).name());
		std::stringstream ss;
		ss << type_str;
		std::string word;
		ss >> word;
		ss >> word;

		return word;
	};
private:

	Problem<T> m_problem;
	Device<T> m_device;
	Loop<T, Device, Kernel> m_looper;

	int m_size;
	std::vector<T> m_in;
	std::vector<T> m_out;
	int m_in_col, m_in_row, m_out_col, m_out_row;

	T* m_device_in;
	T* m_device_out;
};
