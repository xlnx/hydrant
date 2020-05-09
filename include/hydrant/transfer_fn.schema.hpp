#pragma once

#include <vector>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct TransferFnConfig : vm::json::Serializable<TransferFnConfig>
	{
		VM_JSON_FIELD( std::vector<float>, values ) = { 0, 0, 0, 0, 1, 1, 1, 1 };
		VM_JSON_FIELD( std::string, preset ) = "";
	};
}

VM_END_MODULE()
