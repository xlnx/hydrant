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
	struct Texture1DOptions
	{
		VM_DEFINE_ATTRIBUTE( unsigned, length );
		VM_DEFINE_ATTRIBUTE( ConstVoidPtr, data );
		VM_DEFINE_ATTRIBUTE( cufx::Texture::Options, opts );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};

	template <typename T>
	struct Texture1D
	{
		Texture1D() = default;

		Texture1D( Texture1DOptions const &opts ) :
		  opts( opts )
		{
			if ( opts.device.has_value() ) {
				auto arr = opts.device.value().alloc_arraynd<T, 1>( opts.length );
				auto view = cufx::MemoryView1D<T>( (T *)opts.data, opts.length );
				cufx::memory_transfer( arr, view )
				  .launch();
				auto tex = cufx::Texture( arr, opts.opts );
				cuda.reset( new Cuda{ arr, tex, view } );
			} else {
				cpu.reset( new CpuSampler<T>(
				  reinterpret_cast<T const *>( opts.data ),
				  glm::uvec3( opts.length, 0, 0 ),
				  opts.opts ) );
			}
		}

	public:
		Sampler update_sampler()
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

		Sampler sampler() const
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
			cufx::Array1D<T> arr;
			cufx::Texture tex;
			cufx::MemoryView1D<T> view;
		};

	private:
		std::shared_ptr<Cuda> cuda;
		std::shared_ptr<CpuSampler<T>> cpu;
		Texture1DOptions opts;
	};
}

VM_END_MODULE()
