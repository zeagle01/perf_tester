
#pragma once

#include <sstream>
#include <string>

#include "loop.h"



template< typename...P>
struct Extract_Type_String;

template<>
struct Extract_Type_String<>
{
	static inline const std::string value = "";
};

template<typename H,typename ...P>
struct Extract_Type_String<H, P...>
{
	static inline const std::string  local_value = typeid(H).name();
	static inline const std::string  value = local_value + Extract_Type_String< P...>::value;
};


template<
	typename T, 
	template<typename U> typename Problem, 
	template<typename U> typename Device, 
	template<typename U> typename Kernel,
	typename ... Loop_Param
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

		std::string ret;
		auto problem_str = std::string(typeid(Problem<T>).name());
		std::stringstream ss0;
		ss0 << problem_str;
		std::string word;
		ss0 >> word;
		ss0 >> word;
		ret = word;

		auto device_str = std::string(typeid(Device).name());
		std::stringstream ss1;
		ss1 << device_str;

		ss1 >> word;
		ss1 >> word;
		ret += "_at_" + word;

		const auto typenameaa = typeid(int).name();
		auto param_str = std::string(Extract_Type_String<Loop_Param...>::value);
		if (!param_str.empty())
		{
			std::stringstream ss2;
			ss2 << param_str;

			ss2 >> word;
			ss2 >> word;
			ret += "_with_" + word;
		}
		return ret;
	};
private:

	Problem<T> m_problem;
	Device<T> m_device;
	Loop<T, Device, Kernel, Loop_Param...> m_looper;

	int m_size;
	std::vector<T> m_in;
	std::vector<T> m_out;
	int m_in_col, m_in_row, m_out_col, m_out_row;

	T* m_device_in;
	T* m_device_out;
};
