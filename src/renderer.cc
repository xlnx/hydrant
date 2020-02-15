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
			  switch ( cfg.renderer._to_integral() ) {
			  case RendererName::Volume: return new VolumeRenderer;
			  case RendererName::Blocks: return new BlocksRenderer;
			  default:
				  throw std::logic_error( vm::fmt( "unknown renderer '{}'", cfg.renderer ) );
			  }
		  }() );
		vis->init( dataset, cfg );
		return vis;
	}
}

VM_END_MODULE()
