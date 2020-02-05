#pragma once

#include <glm_math.hpp>
#include <VMUtils/modules.hpp>
#include <cudafx/device.hpp>
#include <cudafx/memory.hpp>
#include <cudafx/texture.hpp>
#include <cudafx/transfer.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename T>
	struct Texture3D
	{
		Texture3D( cufx::Device const &device,
				   T *phost,
				   uvec3 const &dim,
				   cufx::Texture::Options const &opts ) :
		  ext( cufx::Extent{}
				 .set_width( dim.x )
				 .set_height( dim.y )
				 .set_depth( dim.z ) ),
		  arr( device.alloc_arraynd<T, 3>( ext ) ),
		  view( phost,
				cufx::MemoryView2DInfo{}
				  .set_stride( dim.x * sizeof( T ) )
				  .set_width( dim.x )
				  .set_height( dim.y ),
				ext ),
		  tex( arr, opts )
		{
			emit().launch();
		}

	public:
		auto emit() const { return cufx::memory_transfer( arr, view ); }
		auto get() const { return tex.get(); }

	private:
		cufx::Extent ext;
		cufx::MemoryView3D<T> view;
		cufx::Array3D<T> arr;
		cufx::Texture tex;
	};
}

VM_END_MODULE()
