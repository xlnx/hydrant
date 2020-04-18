#pragma once

#include <hydrant/basic_renderer.schema.hpp>
#include <hydrant/core/scene.schema.hpp>
#include <hydrant/core/renderer.schema.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct RenderParamConfig : vm::json::Serializable<RenderParamConfig>
	{
		VM_JSON_FIELD( CameraConfig, camera );
		VM_JSON_FIELD( RendererConfig, render );
	};

	struct Config : vm::json::Serializable<Config>
	{
		VM_JSON_FIELD( std::string, data_path );
		VM_JSON_FIELD( RenderParamConfig, params );
	};
}

VM_END_MODULE()

