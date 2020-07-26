

#include "log.h"
#include <string>

#include "profiler.h"


std::shared_ptr<spdlog::logger> Log::s_logger;

void Log::init()
{


	RECORD_FUNCTION_DURATION();

	std::string format = "%^%T:%f %n: %v%$";
	//std::string format = "%+";
	s_logger = spdlog::stdout_color_mt("core");
	s_logger->set_pattern(format);
	s_logger->set_level(spdlog::level::trace);


}

 std::shared_ptr<spdlog::logger>	Log::get_logger()
{
	if (!s_logger)
	{
		init();
	}

	return s_logger;

}

