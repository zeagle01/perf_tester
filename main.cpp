

#include "log.h"
#include "profiler.h"

#include "runner.h"


int main()
{

	BEGIN_PROFILING();


	CE_INFO("begin runing");

	Runner runner;
	runner.run();
	
	END_PROFILING();

	return 0;
}