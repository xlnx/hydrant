#pragma once

#include <string>
#include <fstream>
#include <cppfs/FilePath.h>
#include <VMUtils/enum.hpp>
#include <varch/package_meta.hpp>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/raycaster.hpp>
#include <hydrant/core/render_loop.hpp>

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct Dataset
	{
		VM_DEFINE_ATTRIBUTE( vol::PackageMeta, meta );
		VM_DEFINE_ATTRIBUTE( cppfs::FilePath, root );
	};

	struct RendererConfig : vm::json::Serializable<RendererConfig>
	{
		VM_JSON_FIELD( glm::ivec2, resolution ) = { 512, 512 };
		VM_JSON_FIELD( std::string, renderer );
		VM_JSON_FIELD( vm::json::Any, params ) = vm::json::Any();
	};

	VM_ENUM( RealtimeRenderQuality,
			 Lossless, Dynamic );

	struct RealtimeRenderOptions : vm::json::Serializable<RealtimeRenderOptions>
	{
		VM_JSON_FIELD( RealtimeRenderQuality, quality ) = RealtimeRenderQuality::Dynamic;
	};

	struct IRenderer : vm::Dynamic
	{
		virtual bool init( std::shared_ptr<Dataset> const &dataset, RendererConfig const &cfg )
		{
			this->dataset = dataset;
			this->resolution = cfg.resolution;
			return true;
		}

		virtual void update( vm::json::Any const &params ) {}

		virtual cufx::Image<> offline_render( Camera const &camera ) = 0;

		virtual void realtime_render( IRenderLoop &loop, RealtimeRenderOptions const &opts = RealtimeRenderOptions{} ) = 0;

	protected:
		std::shared_ptr<Dataset> dataset;
		glm::ivec2 resolution;
		Raycaster raycaster;
	};

	struct RendererFactory
	{
		RendererFactory( cppfs::FilePath &dataset_path ) :
		  dataset( new Dataset )
		{
			std::ifstream is( dataset_path.resolve( "package_meta.json" ).resolved() );
			is >> dataset->meta;
			dataset->root = dataset_path;
		}

		vm::Box<IRenderer> create( RendererConfig const &cfg );

		static std::vector<std::string> list_candidates();

	private:
		std::shared_ptr<Dataset> dataset;
	};
}

struct RendererRegistry
{
	static RendererRegistry &instance()
	{
		static RendererRegistry _;
		return _;
	}

	std::map<std::string, std::function<IRenderer *()>> types;
};

#define REGISTER_RENDERER( T, name ) \
	REGISTER_RENDERER_UNIQ_HELPER( __COUNTER__, T, name )

#define REGISTER_RENDERER_UNIQ_HELPER( ctr, T, name ) \
	REGISTER_RENDERER_UNIQ( ctr, T, name )

#define REGISTER_RENDERER_UNIQ( ctr, T, name )                               \
	static int                                                               \
	  renderer_registrar__body__##ctr##__object =                            \
		(                                                                    \
		  ::hydrant::__inner__::RendererRegistry::instance().types[ name ] = \
			[]() -> ::hydrant::IRenderer * { return new T; },                \
		  0 )

VM_END_MODULE()
