


#pragma once
#include <memory>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/fmt/ostr.h"



class Log
{
public:
	static void init();

	static std::shared_ptr<spdlog::logger>get_logger();

private:
	static std::shared_ptr<spdlog::logger> s_logger;
};


#define CE_ERROR(...)	Log::get_logger()->error(__VA_ARGS__)
#define CE_WARN(...)	Log::get_logger()->warn(__VA_ARGS__)
#define CE_INFO(...)	Log::get_logger()->info(__VA_ARGS__)
#define CE_TRACE(...)	Log::get_logger()->trace(__VA_ARGS__)
#define CE_FATAL(...)	Log::get_logger()->fatal(__VA_ARGS__)

