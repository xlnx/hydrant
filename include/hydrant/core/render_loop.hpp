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
		IRenderLoop( Camera const &camera ) :
		  camera( camera )
		{
		}

	public:
		virtual void post_loop() {}

		virtual bool should_stop() = 0;

		virtual void post_frame() {}

		virtual void on_frame( cufx::Image<> &frame ) = 0;

		virtual void after_frame() {}

		virtual void after_loop() {}

	public:
		VM_DEFINE_ATTRIBUTE( Camera, camera );
	};
}

VM_END_MODULE()
