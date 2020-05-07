#pragma once

#include <glog/logging.h>
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
		VM_DEFINE_ATTRIBUTE( cufx::Texture::Options, opts );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
	};

	template <typename T>
	struct Texture1D
	{
	private:
		using NormalizeFloatVec = vec<VecDim<T>::value, float>;
		using CudaVec = typename CudaVecType<T>::type;

	public:
		Texture1D() = default;

		Texture1D( Texture1DOptions const &tex_opts ) :
		  opts( tex_opts )
		{
			if ( opts.device.has_value() ) {
				opts.opts.device = opts.device.value();
				auto arr = opts.device.value().alloc_arraynd<CudaVec, 1>( opts.length );
				cuda.reset( new Cuda{ arr, cufx::Texture( arr, opts.opts ) } );
			} else {
				cpu.reset( new Cpu );
				if ( opts.opts.read_mode == cufx::Texture::ReadMode::NormalizedFloat ) {
					cpu->sampler.reset( new CpuSampler<NormalizeFloatVec>( glm::uvec3( opts.length, 0, 0 ), opts.opts ) );
				} else {
					cpu->sampler.reset( new CpuSampler<T>( glm::uvec3( opts.length, 0, 0 ), opts.opts ) );
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
				cufx::MemoryView1D<T> ptr_view( const_cast<T *>( ptr ), opts.length );
				auto fut = source( ptr_view );
				fut.wait();
			}
		}
		std::future<bool> source( cufx::MemoryView1D<T> const &view )
		{
			std::future<cufx::Result> fut;
			if ( cpu ) {
				if ( need_saturate() ) {
					if ( !cpu->norm_buf.has_value() ) { cpu->norm_buf = std::vector<NormalizeFloatVec>( opts.length ); }
					auto &buf = cpu->norm_buf.value();
					std::transform( view.ptr(), view.ptr() + view.size(), buf.begin(),
									[]( auto val ) { return NormalizeFloatVec( saturate_to_float( val ) ); } );
					static_cast<CpuSampler<NormalizeFloatVec> *>( cpu->sampler.get() )->source( cpu->norm_buf.value().data() );
				} else {
					if ( !cpu->buf.has_value() ) { cpu->buf = std::vector<NormalizeFloatVec>( opts.length ); }
					auto &buf = cpu->buf.value();
					memcpy( buf.data(), view.ptr(), view.size() * sizeof( T ) );
					static_cast<CpuSampler<T> *>( cpu->sampler.get() )->source( cpu->buf.value().data() );
				}
				fut = std::async( std::launch::deferred, [] { return cufx::Result(); } );
			} else {
				fut = cufx::memory_transfer( cuda->arr,
											 reinterpret_cast<cufx::MemoryView1D<CudaVec> const &>( view ) )
						.launch_async();
			}
			return std::async( std::launch::deferred,
							   [&, f = std::move( fut )]() mutable {
								   f.wait();
								   auto res = f.get();
								   if ( !res.ok() ) {
									   LOG( ERROR ) << vm::fmt( "source texture failed: {}", res.message() );
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
				return *cpu->sampler;
			}
		}

	private:
		bool need_saturate() const
		{
			return !std::is_same<T, float>::value &&
				   !std::is_same<T, NormalizeFloatVec>::value &&
				   opts.opts.read_mode == cufx::Texture::ReadMode::NormalizedFloat;
		}

	private:
		struct Cuda
		{
			cufx::Array1D<CudaVec> arr;
			cufx::Texture tex;
		};

		struct Cpu
		{
			std::unique_ptr<ICpuSampler> sampler;
			vm::Option<std::vector<T>> buf;
			vm::Option<std::vector<NormalizeFloatVec>> norm_buf;
		};

	private:
		std::shared_ptr<Cuda> cuda;
		std::shared_ptr<Cpu> cpu;
		Texture1DOptions opts;
	};
}

VM_END_MODULE()
