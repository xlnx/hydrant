#pragma once

#include <VMUtils/nonnull.hpp>
#include <hydrant/core/renderer.hpp>

VM_BEGIN_MODULE( hydrant )

struct ApplicationImpl;

VM_EXPORT
{
	struct Config : vm::json::Serializable<Config>
	{
		VM_JSON_FIELD( CameraConfig, camera );
		VM_JSON_FIELD( RendererConfig, render );
	};

	struct Application
	{
		Application( RendererFactory &factory,
					 std::vector<Config> const &cfgs = {} );
		~Application();

		void run();

	private:
		vm::Box<ApplicationImpl> _;
	};
}

VM_END_MODULE()
