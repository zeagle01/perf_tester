


#pragma once

#include <chrono>
#include <string>
#include <fstream>




#define PROFILING

#ifdef PROFILING

#define FUNCTION_SCOPE_TIMER(function_name) Scope_Timer time##__LINE__(function_name)
#define RECORD_FUNCTION_DURATION() FUNCTION_SCOPE_TIMER(__FUNCSIG__)

#define FUNCTION_SCOPE_TIMER_RETURN(function_name,duration) Scope_Timer time##__LINE__(function_name,duration)
#define RECORD_FUNCTION_DURATION_RETURN(duration) FUNCTION_SCOPE_TIMER_RETURN(__FUNCSIG__,duration)

#define BEGIN_PROFILING() Profiler::get_singleton().begin_session(__FUNCSIG__)
#define END_PROFILING() Profiler::get_singleton().end_session()

#else
#define RECORD_FUNCTION_DURATION() 
#define BEGIN_PROFILING()
#define END_PROFILING()

#endif





	struct Duration
	{
		long begin;
		long end;
		std::string name;
		uint32_t thread_id;
	};


	class Scope_Timer
	{

	public:
		Scope_Timer(std::string name);
		Scope_Timer(std::string name,long* duration);
		~Scope_Timer();

	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_begin;
		std::string m_name;
		long * m_duration;
	};



	class Profiler
	{
	public:
		Profiler() {}

		void begin_session(std::string session_name,std::string output_file="time_line.json");

		void end_session();

		void write_duration_record(const Duration& duration);

		static Profiler& get_singleton();

	private:
		std::ofstream m_os;
		int m_record_count=0;
	};



