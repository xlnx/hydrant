#pragma once

#include <cudafx/device.hpp>
#include <cudafx/array.hpp>
#include <cudafx/texture.hpp>
#include <cudafx/transfer.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/bridge/buffer3d.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct Texture3DOptions
	{
		VM_DEFINE_ATTRIBUTE( glm::uvec3, dim );
		VM_DEFINE_ATTRIBUTE( cufx::Texture::Options, opts );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};

	template <typename T>
	struct Texture3D
	{
		Texture3D() = default;

		Texture3D( Texture3DOptions const &opts ) :
		  opts( opts )
		{
			if ( opts.device.has_value() ) {
				auto arr = opts.device.value()
							 .alloc_arraynd<T, 3>( cufx::Extent{}
													 .set_width( opts.dim.x )
													 .set_height( opts.dim.y )
													 .set_depth( opts.dim.z ) );
				cuda.reset( new Cuda{ arr, cufx::Texture( arr, opts.opts ) } );
			} else {
				cpu.reset( new Cpu );
				if ( opts.opts.read_mode == cufx::Texture::ReadMode::NormalizedFloat ) {
					cpu->sampler.reset( new CpuSampler<float>( opts.dim, opts.opts ) );
				} else {
					cpu->sampler.reset( new CpuSampler<T>( opts.dim, opts.opts ) );
				}
			}
		}

	public:
		void source( T const *ptr, bool temporary = true )
		{
			if ( !temporary && cpu && !need_saturate() ) {
				static_cast<CpuSampler<T> *>( cpu->sampler.get() )->source( ptr );
				if ( cpu->buf ) cpu->buf = vm::None{};
			} else {
				cufx::MemoryView3D<T> ptr_view( const_cast<T *>( ptr ),
												cufx::MemoryView2DInfo{}
												  .set_stride( opts.dim.x * sizeof( T ) )
												  .set_width( opts.dim.x )
												  .set_height( opts.dim.y ),
												cufx::Extent{}
												  .set_width( opts.dim.x )
												  .set_height( opts.dim.y )
												  .set_depth( opts.dim.z ) );
				auto fut = source( ptr_view );
				fut.wait();
			}
		}
		std::future<bool> source( cufx::MemoryView3D<T> const &view )
		{
			std::future<cufx::Result> fut;
			if ( cpu ) {
				if ( need_saturate() ) {
					if ( !cpu->norm_buf.has_value() ) { cpu->norm_buf = HostBuffer3D<float>( opts.dim ); }
					auto &buf = cpu->norm_buf.value();
					buf.iterate_3d(
					  [&]( auto idx ) {
						  auto val = view.at( idx.x, idx.y, idx.z );
						  buf[ idx ] = saturate_to_float( vec<1, unsigned char>( val ) ).x;
					  } );
					static_cast<CpuSampler<float> *>( cpu->sampler.get() )->source( cpu->norm_buf.value().data() );
				} else {
					if ( !cpu->buf.has_value() ) { cpu->buf = HostBuffer3D<T>( opts.dim ); }
					auto &buf = cpu->buf.value();
					buf.iterate_3d(
					  [&]( auto idx ) {
						  buf[ idx ] = view.at( idx.x, idx.y, idx.z );
					  } );
					static_cast<CpuSampler<T> *>( cpu->sampler.get() )->source( cpu->buf.value().data() );
				}
				fut = std::async( std::launch::deferred, [] { return cufx::Result(); } );
			} else {
				fut = cufx::memory_transfer( cuda->arr, view ).launch_async();
			}
			return std::async( std::launch::deferred,
							   [&, f = std::move( fut )]() mutable {
								   f.wait();
								   auto res = f.get();
								   if ( !res.ok() ) {
									   vm::eprintln( "source texture failed: {}", res.message() );
									   return false;
								   }
								   if ( cuda ) { cuda->tex = cufx::Texture( cuda->arr, opts.opts ); }
								   return true;
							   } );
		}

		Sampler sampler() const
		{
			if ( cuda ) {
				return cuda->tex;
			} else {
				return *cpu->sampler.get();
			}
		}

	private:
		bool need_saturate() const
		{
			return !std::is_same<T, float>::value &&
				   opts.opts.read_mode == cufx::Texture::ReadMode::NormalizedFloat;
		}

	private:
		struct Cuda
		{
			cufx::Array3D<T> arr;
			cufx::Texture tex;
		};
		struct Cpu
		{
			std::unique_ptr<ICpuSampler> sampler;
			vm::Option<HostBuffer3D<T>> buf;
			vm::Option<HostBuffer3D<float>> norm_buf;
		};

	private:
		std::shared_ptr<Cuda> cuda;
		std::shared_ptr<Cpu> cpu;
		Texture3DOptions opts;
	};
}

VM_END_MODULE()
