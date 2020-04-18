#pragma once

#include <vector>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct TransferFnConfig : vm::json::Serializable<TransferFnConfig>
	{
		VM_JSON_FIELD( std::vector<vec4>, values );
	};
}

VM_END_MODULE()
