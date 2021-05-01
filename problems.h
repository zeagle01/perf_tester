
#pragma once


template<typename T>
struct Vector_Add1
{
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

};

//template<typename T, int N>
//struct Multiply_Add_N_Times
//{
//	void init(std::vector<T>& in, std::vector<T>& out, int& m_in_col, int& m_in_row, int& m_out_col, int& m_out_row, int size)
//	{
//		m_in_row = 2;
//		m_in_col = size;
//		m_out_col = size;
//		m_out_row = 1;
//
//		in.resize(size * 2);
//		out.resize(size);
//	}
//	bool verify(const std::vector<T>& in, const std::vector<T>& out)
//	{
//		int n = out.size();
//		for (int i = 0; i < n; i++)
//		{
//			const T* in0 = &in[0];
//			const T* in1 = &in[n];
//			if (out[i] != in0[i] + in1[i])
//			{
//				return false;
//				break;
//			}
//		}
//		return true;
//	}
//
//	size_t get_problem_size(int size)
//	{
//		return sizeof(T) * size;
//	}
//
//};
