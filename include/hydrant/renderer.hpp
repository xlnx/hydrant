#pragma once

#include <string>
#include <fstream>
#include <hydrant/core/glm_math.hpp>
#include <hydrant/core/raycaster.hpp>
#include "dataset.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	struct RendererConfig : vm::json::Serializable<RendererConfig>
	{
		VM_JSON_FIELD( glm::ivec2, resolution ) = { 512, 512 };
		VM_JSON_FIELD( std::string, renderer ) = "volume";
		VM_JSON_FIELD( vm::json::Any, params ) = vm::json::Any();
	};

	struct Renderer : vm::Dynamic
	{
		virtual bool init( std::shared_ptr<Dataset> const &dataset, RendererConfig const &cfg )
		{
			this->dataset = dataset;
			this->resolution = resolution;
			return true;
		}

		virtual void offline_render( std::string const &dst_path, Camera const &camera ) = 0;

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

		vm::Box<Renderer> create( RendererConfig const &cfg );

	private:
		std::shared_ptr<Dataset> dataset;
	};
}

VM_END_MODULE()
