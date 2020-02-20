#pragma once

#include <varch/utils/io.hpp>
#include <hydrant/unarchiver.hpp>
#include <hydrant/bridge/buffer3d.hpp>
#include <hydrant/basic_renderer.hpp>
#include <hydrant/transfer_fn.hpp>
#include "volume_shader.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct VolumeRendererConfig : vm::json::Serializable<VolumeRendererConfig>
	{
		VM_JSON_FIELD( TransferFnConfig, transfer_fn );
		VM_JSON_FIELD( float, density ) = 1e-2f;
		VM_JSON_FIELD( std::size_t, mem_limit_mb ) = 1024 * 2;
	};

	struct VolumeRenderer : BasicRenderer<VolumeShader>
	{
		using Super = BasicRenderer<VolumeShader>;

		bool init( std::shared_ptr<Dataset> const &dataset,
				   RendererConfig const &cfg ) override;

		cufx::Image<> offline_render( Camera const &camera ) override;

		void render_loop( IRenderLoop &loop ) override;

	private:
		vol::MtArchive const *sample_level( std::size_t level ) const;

		std::shared_ptr<IBuffer3D<unsigned char>> alloc_block_buf( std::size_t pad_bs );

		Texture3DOptions block_tex_opts( std::size_t pad_bs ) const;

		struct LowestLevelBlock
		{
			VM_DEFINE_ATTRIBUTE( vol::Idx, idx );
			VM_DEFINE_ATTRIBUTE( Texture3D<unsigned char>, storage );
			VM_DEFINE_ATTRIBUTE( BlockSampler, sampler );
		};

		std::vector<LowestLevelBlock> unarchive_lowest_level();

	private:
		std::shared_ptr<Unarchiver> uu;

		TransferFn transfer_fn;

		ThumbnailTexture<int> chebyshev;

		Texture3D<int> vaddr;
		HostBuffer3D<int> vaddr_buf;

		std::vector<vol::Idx> block_idxs;
		std::vector<glm::vec3> block_ccs;
		std::vector<int> pidx;
	};
}

VM_END_MODULE()
