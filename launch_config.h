
#pragma once

template<int N>
struct Launch_Config
{
	static constexpr int thread_per_block = N;
};
