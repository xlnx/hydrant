#pragma once

#include <cudafx/device.hpp>
#include <cudafx/array.hpp>
#include <cudafx/texture.hpp>
#include <cudafx/transfer.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/bridge/sampler.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct ConstTexture3DOptions
	{
		using ConstVoidPtr = void const *;

		VM_DEFINE_ATTRIBUTE( glm::uvec3, dim );
		VM_DEFINE_ATTRIBUTE( ConstVoidPtr, data );
		VM_DEFINE_ATTRIBUTE( cufx::Texture::Options, opts );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};

	template <typename T>
	struct ConstTexture3D
	{
		ConstTexture3D() = default;

		ConstTexture3D( ConstTexture3DOptions const &opts ) :
		  opts( opts )
		{
			if ( opts.device.has_value() ) {
				auto extent = cufx::Extent{}
								.set_width( opts.dim.x )
								.set_height( opts.dim.y )
								.set_depth( opts.dim.z );
				auto view_info = cufx::MemoryView2DInfo{}
								   .set_stride( opts.dim.x * sizeof( T ) )
								   .set_width( opts.dim.x )
								   .set_height( opts.dim.y );
				auto arr = opts.device.value().alloc_arraynd<T, 3>( extent );
				auto view = cufx::MemoryView3D<T>( (T *)opts.data, view_info, extent );
				cufx::memory_transfer( arr, view )
				  .launch();
				auto tex = cufx::Texture( arr, opts.opts );
				cuda.reset( new Cuda{ arr, tex, view } );
			} else {
				cpu.reset( new CpuSampler<T>(
				  reinterpret_cast<T const *>( opts.data ),
				  opts.dim,
				  opts.opts ) );
			}
		}

	public:
		Sampler update()
		{
			if ( cuda ) {
				cufx::memory_transfer( cuda->arr, cuda->view )
				  .launch();
				cuda->tex = cufx::Texture( cuda->arr, opts.opts );
				return cuda->tex;
			} else {
				return cpu.get();
			}
		}

		Sampler get() const
		{
			if ( cuda ) {
				return cuda->tex;
			} else {
				return cpu.get();
			}
		}

	private:
		struct Cuda
		{
			cufx::Array3D<T> arr;
			cufx::Texture tex;
			cufx::MemoryView3D<T> view;
		};

	private:
		std::shared_ptr<Cuda> cuda;
		std::shared_ptr<CpuSampler<T>> cpu;
		ConstTexture3DOptions opts;
	};
}

VM_END_MODULE()
