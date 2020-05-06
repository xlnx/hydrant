#pragma once

#include <VMUtils/enum.hpp>
#include <VMUtils/json_binding.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct RendererConfig : vm::json::Serializable<RendererConfig>
	{
		VM_JSON_FIELD( glm::ivec2, resolution ) = { 512, 512 };
		VM_JSON_FIELD( std::string, renderer );
		VM_JSON_FIELD( vm::json::Any, params ) = vm::json::Any();
	};

	VM_ENUM( RealtimeRenderQuality,
			 Lossless, Dynamic );
}

VM_END_MODULE()
