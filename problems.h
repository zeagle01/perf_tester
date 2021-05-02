
#pragma once
#include "compute_kernels.h"
#include "type_list.h"
#include <vector>


template<typename T>
struct Vector_Add 
{
	using param_list = type_list<>;
	using Kernel = Add_Kernel<T>;
	using Data_Type = T;

	void init(std::vector<T>& in, std::vector<T>& out, int& m_in_col, int& m_in_row, int& m_out_col, int& m_out_row, int size)
	{
		m_in_row = 2;
		m_in_col = size;
		m_out_col = size;
		m_out_row = 1;

		in.resize(size * 2);
		out.resize(size);
		T* in0 = &in[0];
		T* in1 = &in[size];
		for (int i = 0; i < size; i++)
		{
			in0[i] = 1;
			in1[i] = 1;
			out[i] = 0;
		}
	}
	bool verify(const std::vector<T>& in, const std::vector<T>& out)
	{
		int n = out.size();
		for (int i = 0; i < n; i++)
		{
			const T* in0 = &in[0];
			const T* in1 = &in[n];
			if (out[i] != in0[i] + in1[i])
			{
				return false;
				break;
			}
		}
		return true;
	}

	size_t get_problem_size(int size)
	{
		return sizeof(T) * size;
	}

	size_t get_operation_size(int size)
	{
		return get_problem_size(size);
	}

};






template<int N>
struct Repeat 
{
	static constexpr int value = N;
};



template<typename T, typename repeat_count>
struct Multiply_Add_N_Times
{

	using param_list = type_list<repeat_count>;
	using Kernel = Mupltiply_Add_N_Times_Kernel<T, repeat_count>;
	using Data_Type = T;

	void init(std::vector<T>& in, std::vector<T>& out, int& m_in_col, int& m_in_row, int& m_out_col, int& m_out_row, int size)
	{
		m_in_row = 2;
		m_in_col = size;
		m_out_col = size;
		m_out_row = 1;

		in.resize(size * 2);
		out.resize(size);
		T* in0 = &in[0];
		T* in1 = &in[size];
		for (int i = 0; i < size; i++)
		{
			in0[i] = 0.9;
			in1[i] = 1;
			out[i] = 0;
		}
	}

	bool verify(const std::vector<T>& in, const std::vector<T>& out)
	{
		int n = out.size();
		for (int i = 0; i < n; i++)
		{
			if (std::abs(out[i] - 10.) > 1e-3)
			{
				return false;
				break;
			}
		}
		return true;
	}

	size_t get_problem_size(int size)
	{
		return sizeof(T) * size;
	}

	size_t get_operation_size(int size)
	{
		return get_problem_size(size) * repeat_count::value;
	}

};



template<int N>
struct Neighbor_Width 
{
	static constexpr int value = N;
};

template<typename T, typename neighbor_width>
struct Convolution
{

	using param_list = type_list<neighbor_width>;
	using Kernel = Convolution_Kernel<T, neighbor_width>;
	using Data_Type = T;

	void init(std::vector<T>& in, std::vector<T>& out, int& m_in_col, int& m_in_row, int& m_out_col, int& m_out_row, int size)
	{
		m_in_row = 1;
		m_in_col = size + 2 * neighbor_width::value;
		m_out_col = size;
		m_out_row = 1;

		in.resize(m_in_col, 1);
		out.resize(size, 0);
	}

	bool verify(const std::vector<T>& in, const std::vector<T>& out)
	{
		int n = out.size();
		for (int i = 0; i < n; i++)
		{
			if (std::abs(out[i] - 1.) > 1e-3)
			{
				return false;
				break;
			}
		}
		return true;
	}

	size_t get_problem_size(int size)
	{
		return sizeof(T) * size;
	}

	size_t get_operation_size(int size)
	{
		return get_problem_size(size) * neighbor_width::value * 2;
	}

};



