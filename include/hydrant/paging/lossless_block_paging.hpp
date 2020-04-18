#pragma once

#include <cudafx/device.hpp>
#include <cudafx/texture.hpp>
#include <hydrant/core/renderer.hpp>
#include <hydrant/bridge/sampler.hpp>
#include <hydrant/octree_culler.hpp>
#include <hydrant/paging/block_paging.hpp>

VM_BEGIN_MODULE( hydrant )

struct LosslessBlockPagingServerImpl;

VM_EXPORT
{
	struct LosslessBlockPagingServer;

	struct LosslessBlockPagingServerOptions
	{
		VM_DEFINE_ATTRIBUTE( std::size_t, mem_limit_mb ) = 1024 * 2;
		VM_DEFINE_ATTRIBUTE( std::shared_ptr<Dataset>, dataset );
		VM_DEFINE_ATTRIBUTE( vm::Option<cufx::Device>, device );
		VM_DEFINE_ATTRIBUTE( cufx::Texture::Options, storage_opts );
	};

	struct LosslessBlockPagingState
	{
		bool next( BlockPaging &paging );

	private:
		int i = 0;
		std::vector<vol::Idx> block_idxs;
		std::vector<int> pidx;
		LosslessBlockPagingServerImpl *self;
		friend struct LosslessBlockPagingServer;
	};

	struct LosslessBlockPagingServer : vm::NoCopy, vm::NoMove
	{
		LosslessBlockPagingServer( LosslessBlockPagingServerOptions const &opts );
		~LosslessBlockPagingServer();

	public:
		LosslessBlockPagingState start( OctreeCuller &culler,
										Camera const &camera,
										mat4 const &et );

	private:
		std::unique_ptr<LosslessBlockPagingServerImpl> _;
	};
}

VM_END_MODULE()
