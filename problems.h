
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


template<typename T, template<typename >typename Kt>
struct Laplician_1D_Template
{
	using param_list = type_list<>;
	using Kernel = Kt<T>;
	using Data_Type = T;
	using Kernel_Prameter_Type =  typename Kernel::template Parameter_Type<T>;

	void init_host(int size)
	{
		m_size = size;

		m_A.push_back(std::vector<T>());
		auto& row_0_of_A = m_A.back();
		row_0_of_A.push_back(2);
		row_0_of_A.push_back(-1);

		m_column_index.push_back(std::vector<int>());
		auto& row_0_of_column_index = m_column_index.back();
		row_0_of_column_index.push_back(0);
		row_0_of_column_index.push_back(1);

		for (int i = 1; i < size-1; i++)
		{
			m_A.push_back(std::vector<T>());
			auto& row_i_of_A = m_A.back();
			row_i_of_A.push_back(2); //put diagonal first
			row_i_of_A.push_back(-1);
			row_i_of_A.push_back(-1);

			m_column_index.push_back(std::vector<int>());
			auto& row_i_of_column_index = m_column_index.back();
			row_i_of_column_index.push_back(i);
			row_i_of_column_index.push_back(i-1);
			row_i_of_column_index.push_back(i + 1);
		}

		m_A.push_back(std::vector<T>());
		auto& row_n_of_A = m_A.back();
		row_n_of_A.push_back(2);
		row_n_of_A.push_back(-1);

		m_column_index.push_back(std::vector<int>());
		auto& row_n_of_column_index = m_column_index.back();
		row_n_of_column_index.push_back(size - 1);
		row_n_of_column_index.push_back(size - 2);


		m_b.assign(size, 0);
		m_x0.assign(size, 1);
		m_x.assign(size, 1);
	}


	bool verify(const Kernel_Prameter_Type& param)
	{
		std::vector<float> exp(m_size, 0.f);
		exp[0] = 1.f;
		exp[m_size - 1] = 1.f;
		for (int i = 0; i < m_size; i++)
		{
			if (std::abs(param.x[i] - exp[i]) > 1e-6f)
			{
				return false;
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
		return get_problem_size(size) * 3;
	}


	std::vector<std::vector<T>> m_A;
	std::vector<std::vector<int>> m_column_index;
	std::vector<T> m_b;
	std::vector<T> m_x;
	std::vector<T> m_x0;
	int m_size;

};

template<typename T, template<typename >typename Kt>
struct Laplician_1D;

template<typename T>
struct Laplician_1D<T, Matrix_Vector_Multiplication_ELL> : Laplician_1D_Template<T, Matrix_Vector_Multiplication_ELL>
{
	void init(ELL_Matrix_And_Vector<T>& param, int size)
	{

		using Base = Laplician_1D_Template<T, Matrix_Vector_Multiplication_ELL>;

		Base::init_host( size);

		m_ell_A.assign(size * 3, 0);
		m_ell_J.assign(size * 4, 0);
		for (int i = 0; i < size; i++)
		{
			m_ell_J[0 * size + i] = Base::m_A[i].size();
			for (int j = 0; j < Base::m_A[i].size(); j++)
			{
				m_ell_A[j * size + i] = Base::m_A[i][j];
				m_ell_J[(j + 1) * size + i] = Base::m_column_index[i][j];
			}
		}
		param.A.data = m_ell_A.data();
		param.A.count = 3 * size;
		param.A.col = size;

		param.J.data = m_ell_J.data();
		param.J.count = 4 * size;
		param.J.col = size;

		param.b = Base::m_b;
		param.x = Base::m_x;
		param.x0 = Base::m_x0;
	}
	std::vector<T> m_ell_A;
	std::vector<int> m_ell_J;
};



template<typename T>
struct Laplician_1D<T, Matrix_Vector_Multiplication_CSR> : Laplician_1D_Template<T, Matrix_Vector_Multiplication_CSR>
{
	void init(CSR_Matrix_And_Vector<T>& param, int size)
	{
		using Base = Laplician_1D_Template<T, Matrix_Vector_Multiplication_CSR>;

		Base::init_host(size);
		m_CSR_I.resize(size + 1);
		m_CSR_I[0] = 0;
		for (int i = 0; i < Base::m_A.size(); i++)
		{
			m_CSR_I[i + 1] += m_CSR_I[i] + Base::m_A[i].size();
			for (int j = 0; j < Base::m_A[i].size(); j++)
			{
				m_CSR_A.push_back(Base::m_A[i][j]);
				m_CSR_J.push_back(Base::m_column_index[i][j]);
			}
		}

		param.A = m_CSR_A;
		param.I = m_CSR_I;
		param.J = m_CSR_J;

		param.b = Base::m_b;
		param.x = Base::m_x;
		param.x0 = Base::m_x0;
	}

	std::vector<T> m_CSR_A;
	std::vector<int> m_CSR_I;
	std::vector<int> m_CSR_J;
};



