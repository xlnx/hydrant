#pragma once

#include <cudafx/device.hpp>
#include <cudafx/texture.hpp>
#include <hydrant/core/renderer.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/octree_culler.hpp>

VM_BEGIN_MODULE( hydrant )

struct RtBlockPagingServerImpl;

VM_EXPORT
{
	struct BlockSamplerMapping
	{
		__host__ __device__ vec3
		  mapped( vec3 const &x ) const
		{
			return k * x + b;
		}

	public:
		VM_DEFINE_ATTRIBUTE( float, k );
		VM_DEFINE_ATTRIBUTE( vec3, b );
	};

	struct BlockSampler
	{
		template <typename T>
		__host__ __device__ T
		  sample_3d( vec3 const &x ) const
		{
			return sampler.sample_3d<T>( mapping.mapped( x ) );
		}

	public:
		VM_DEFINE_ATTRIBUTE( Sampler, sampler );
		VM_DEFINE_ATTRIBUTE( BlockSamplerMapping, mapping );
	};

	struct RtBlockPagingClient
	{
	public:
		Sampler vaddr;
		int lowest_blkcnt;
		BlockSampler const *block_sampler;
	};

	struct RtBlockPagingServerOptions
	{
		VM_DEFINE_ATTRIBUTE( uvec3, dim );
		VM_DEFINE_ATTRIBUTE( std::size_t, mem_limit_mb ) = 1024 * 2;
		VM_DEFINE_ATTRIBUTE( std::shared_ptr<Dataset>, dataset );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
		VM_DEFINE_ATTRIBUTE( cufx::Texture::Options, storage_opts );
	};

	struct RtBlockPagingServer : vm::NoCopy, vm::NoMove
	{
		RtBlockPagingServer( RtBlockPagingServerOptions const &opts );
		~RtBlockPagingServer();

	public:
		RtBlockPagingClient update( OctreeCuller &culler, Camera const &camera );

		void start();

		void stop();

	private:
		std::unique_ptr<RtBlockPagingServerImpl> _;
	};
}

VM_END_MODULE()
