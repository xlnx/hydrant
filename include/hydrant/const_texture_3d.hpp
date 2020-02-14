#pragma once

#include <cudafx/device.hpp>
#include <cudafx/array.hpp>
#include <cudafx/texture.hpp>
#include <cudafx/transfer.hpp>
#include <hydrant/core/glm_math.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	template <typename T>
	struct ConstTexture3D
	{
		ConstTexture3D( glm::uvec3 const &dim, T const *data,
						cufx::Texture::Options const &opts, cufx::Device const &device ) :
		  opts( opts )
		{
			auto extent = cufx::Extent{}
							.set_width( dim.x )
							.set_height( dim.y )
							.set_depth( dim.z );
			auto view_info = cufx::MemoryView2DInfo{}
							   .set_stride( dim.x * sizeof( T ) )
							   .set_width( dim.x )
							   .set_height( dim.y );
			view = cufx::MemoryView3D<T>( const_cast<T *>( data ), view_info, extent );
			arr = device.alloc_arraynd<T, 3>( extent );
			update();
		}

	public:
		cufx::Texture const &update()
		{
			cufx::memory_transfer( arr.value(), view )
			  .launch();
			tex = cufx::Texture( arr.value(), opts );
			return tex.value();
		}

		cufx::Texture const &get() const { return tex.value(); }

	private:
		vm::Option<cufx::Array3D<T>> arr;
		vm::Option<cufx::Texture> tex;
		cufx::Texture::Options opts;
		cufx::MemoryView3D<T> view;
	};
}

VM_END_MODULE()
