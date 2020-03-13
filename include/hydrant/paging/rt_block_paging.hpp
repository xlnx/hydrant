#pragma once

#include <cudafx/device.hpp>
#include <cudafx/texture.hpp>
#include <hydrant/core/renderer.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/octree_culler.hpp>
#include <hydrant/paging/block_paging.hpp>

VM_BEGIN_MODULE( hydrant )

struct RtBlockPagingServerImpl;

VM_EXPORT
{
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
		BlockPaging update( OctreeCuller &culler, Camera const &camera );

		void start();

		void stop();

	private:
		std::unique_ptr<RtBlockPagingServerImpl> _;
	};
}

VM_END_MODULE()
