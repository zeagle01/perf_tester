
#pragma once

#include <string>
#include <vector>







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



