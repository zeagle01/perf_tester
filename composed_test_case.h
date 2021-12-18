
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
		m_problem.init(m_kernel_parameter_host, size);
		m_device.upload(m_kernel_parameter_device, m_kernel_parameter_host);
		m_looper.before_loop(m_kernel_parameter_device, m_kernel_parameter_host, size);
	};
	void run() override 
	{ 
		m_looper.apply(m_kernel_parameter_device, m_size);
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
		m_device.dowload(m_kernel_parameter_host, m_kernel_parameter_device);
		auto ret = m_problem.verify(m_kernel_parameter_host);
		m_device.free_device_source(m_kernel_parameter_device);
		return ret;
	};
	std::string get_name() override 
	{ 
		auto problem_type_str = std::string(typeid(Problem).name());
		auto problem_str = strip_type_string(problem_type_str, "\\s+|<|>");

		auto problem_template_str = extrext_template(problem_type_str).back();

		auto problem_param_type_str = Extract_Name_Of_Type_List<Problem::param_list>::value;
		auto problem_param_str = strip_type_string(problem_param_type_str, "\\s+");

		auto device_type_str = std::string(typeid(Device).name());
		auto device_str = strip_type_string(device_type_str, "\\s+|<|>");

		auto device_param_type_str = Extract_Name_Of_Type_List<Device::param_list>::value;
		auto device_param_str = strip_type_string(device_param_type_str, "\\s+");

		auto data_type_str = std::string(typeid(T).name());


		std::string ret = problem_str + "<" + data_type_str + "," + problem_template_str + "," + problem_param_str + ">_" + device_str + "<" + device_param_str + ">";
		return ret;
	};


private:
	std::vector<std::string> extrext_template(const std::string& type_string)
	{
		std::vector<std::string> ret;

		ret = re_search(type_string, "<.*>");
		if (!ret.empty())
		{
			ret = strip_type_string_vector(ret[0], "\\s+|,|<|>");
		}
		return ret;
	}

	std::vector<std::string> re_search(const std::string& type_string, const std::string& re_pattern)
	{
		std::vector<std::string> ret;

		std::regex rgx(re_pattern);
		std::smatch m;
		std::regex_search(type_string, m, rgx);
		for (auto x : m)
		{
			ret.push_back(x);
		}
		return ret;
	}

	std::vector<std::string> strip_type_string_vector(const std::string& type_string,std::string delimeter)
	{
		std::vector<std::string> ret;
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
				ret.push_back(cc);
			}
		}
		return ret;
	}

	std::string strip_type_string(const std::string& type_string,std::string delimeter)
	{
		std::string ret;
		if (type_string.empty())
		{
			return ret;
		}
		auto v = strip_type_string_vector(type_string, delimeter);
		if (v.empty())
		{
			return ret;
		}
		else
		{
			return v[0];
		}
	}
private:

	Problem m_problem;
	using T = typename Problem::Data_Type;
	using Kernel_Type = typename  Problem::Kernel;
	template<typename U>
	using Kernel_Parameter_Type = typename  Kernel_Type::template Parameter_Type<U>;

	using Device_T = typename Device::template type<T, Kernel_Parameter_Type>;
	Device_T m_device;

	typename Device_T::template Looper< Kernel_Type > m_looper;
	
	int m_size;

	Kernel_Parameter_Type<T> m_kernel_parameter_device;
	Kernel_Parameter_Type<T> m_kernel_parameter_host;
};
