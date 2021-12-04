
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

	void init(Matrix_In_Matrix_Out<T>& param, int size)
	{
		param.in_row = 2;
		param.in_col = size;

		param.out_row = 1;
		param.out_col = size;

		m_in.resize(size * 2);
		m_out.resize(size);
		param.in_data= m_in;
		param.out_data = m_out;

		T* in0 = &param.in_data.data[0];
		T* in1 = &param.in_data.data[size];
		for (int i = 0; i < size; i++)
		{
			in0[i] = 1;
			in1[i] = 1;
			param.out_data.data[i] = 0;
		}

	}

	bool verify(const Matrix_In_Matrix_Out<T>& param)
	{
		int n = param.out_row * param.out_col;
		const T* out = param.out_data;
		const T* in0 = &param.in_data[0];
		const T* in1 = &param.in_data[n];
		for (int i = 0; i < n; i++)
		{
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

private:
	std::vector<T> m_in;
	std::vector<T> m_out;

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

	void init(Matrix_In_Matrix_Out<T>& param, int size)
	{
		param.in_row = 2;
		param.in_col = size;

		param.out_row = 1;
		param.out_col = size;

		m_in.resize(size * 2);
		m_out.resize(size);
		param.in_data = m_in;
		param.out_data = m_out;

		T* in0 = &param.in_data[0];
		T* in1 = &param.in_data[size];
		for (int i = 0; i < size; i++)
		{
			in0[i] = 0.9;
			in1[i] = 1;
			param.out_data[i] = 0;
		}
	}


	bool verify(const Matrix_In_Matrix_Out<T>& param)
	{
		int n = param.out_row * param.out_col;
		const T* out = param.out_data;
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

private:
	std::vector<T> m_in;
	std::vector<T> m_out;
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

	void init(Matrix_In_Matrix_Out<T>& param, int size)
	{
		param.in_row = 1;
		param.in_col = size + 2 * neighbor_width::value;

		param.out_row = 1;
		param.out_col = size;

		m_in.resize(param.in_col);
		m_out.resize(size);

		m_in.assign(m_in.size(), 1);
		m_out.assign(m_out.size(), 0);

		param.in_data = m_in;
		param.out_data = m_out;


	}

	bool verify(const Matrix_In_Matrix_Out<T>& param)
	{
		int n = param.out_row * param.out_col;
		const T* out = param.out_data;
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

private:
	std::vector<T> m_in;
	std::vector<T> m_out;

};



