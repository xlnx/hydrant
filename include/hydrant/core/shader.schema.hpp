#pragma once

#include <VMUtils/enum.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	VM_ENUM( ShadingDevice,
			 Cpu, Cuda );

	VM_ENUM( ShadingResult,
			 Ok, Err );
}

VM_END_MODULE()
