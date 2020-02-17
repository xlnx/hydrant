#pragma once

#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include <cudafx/image.hpp>
#include <hydrant/core/scene.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct IRenderLoop : vm::NoCopy, vm::NoMove, vm::Dynamic
	{
		virtual bool should_stop() = 0;

		virtual void post_frame() = 0;

		virtual void on_frame( cufx::Image<> &frame ) = 0;

	public:
		VM_DEFINE_ATTRIBUTE( Camera, camera );
	};
}

VM_END_MODULE()
