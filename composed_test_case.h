
#pragma once

#include <sstream>
#include <string>
#include <regex>

#include "loop.h"
#include "types_helper.h"







template< typename Problem, typename Device >
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
		m_looper.apply(m_device_in, m_device_out, m_size, m_in_col, m_in_row, m_out_col, m_out_row);
	};

	//for async test case
	void sync_wait() override 
	{
		m_device.sync_wait(); 
	}

	size_t get_operation_size_with_respect_to_byte() override 
	{
		return m_problem.get_operation_size(m_size);
	}

	size_t get_problem_size_in_byte() override 
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

		auto problem_type_str = std::string(typeid(Problem).name());
		auto problem_str = strip_type_string(problem_type_str, "\\s+|<|>");

		auto problem_param_type_str = Extract_Name_Of_Type_List<Problem::param_list>::value;
		auto problem_param_str = strip_type_string(problem_param_type_str, "\\s+");

		auto device_type_str = std::string(typeid(Device).name());
		auto device_str = strip_type_string(device_type_str, "\\s+|<|>");

		auto device_param_type_str = Extract_Name_Of_Type_List<Device::param_list>::value;
		auto device_param_str = strip_type_string(device_param_type_str, "\\s+");

		auto data_type_str = std::string(typeid(T).name());

		std::string ret = problem_str + "<" + data_type_str + "," + problem_param_str + ">_" + device_str + "<" + device_param_str + ">";
		return ret;
	};


private:
	std::string strip_type_string(const std::string& type_string,std::string delimeter)
	{
		std::string ret;
		if (type_string.empty())
		{
			return ret;
		}

		std::regex rgx(delimeter);
		std::sregex_token_iterator it(type_string.begin(), type_string.end(), rgx, -1);
		std::sregex_token_iterator end;
		for (; it != end; ++it)
		{
			std::string cc = std::string(*it);
			if (cc != "struct")
			{
				ret.append(cc);
				return ret;
			}
		}
		return ret;
	}
private:

	Problem m_problem;
	using T = typename Problem::Data_Type;

	using Device_T = typename Device::template type<T>;
	Device_T m_device;

	using Problem_Kernel = typename Problem::Kernel;
	typename Device_T::template Looper<Problem_Kernel> m_looper;
	
	int m_size;
	std::vector<T> m_in;
	std::vector<T> m_out;
	int m_in_col, m_in_row, m_out_col, m_out_row;

	T* m_device_in;
	T* m_device_out;
};
