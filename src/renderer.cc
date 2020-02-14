#include <hydrant/renderer.hpp>
#include "renderers/volume/volume.hpp"
#include "renderers/blocks/blocks.hpp"

VM_BEGIN_MODULE( hydrant )

VM_EXPORT
{
	vm::Box<Renderer> RendererFactory::create( RendererConfig const &cfg )
	{
		vm::Box<Renderer> vis(
		  [&]() -> Renderer * {
			  if ( cfg.renderer == "volume" ) return new VolumeRenderer;
			  if ( cfg.renderer == "blocks" ) return new BlocksRenderer;
			  throw std::logic_error( vm::fmt( "unknown renderer '{}'", cfg.renderer ) );
		  }() );
		vis->init( dataset, cfg );
		return vis;
	}
}

VM_END_MODULE()
